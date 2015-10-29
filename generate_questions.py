# Copyright 2014 Google Inc. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script for downloading and generating question/answer pairs.
"""

import argparse
from collections import namedtuple
import hashlib
from itertools import chain
from itertools import izip
from itertools import repeat
import math
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
import os
import re
import sys
import time
import cchardet as chardet
from lxml import html
import requests
import socket


class Story(namedtuple('StoryBase', 'url content highlights')):

  def ToString(self):
    return self.content + ''.join([
        '\n\n@highlight\n\n' + highlight
        for highlight in
        self.highlights])


AnonymizedStory = namedtuple(
    'AnonymizedStory', 'url content highlights anonymization_info')
RawStory = namedtuple('RawStory', 'url html')
TokenizedStory = namedtuple('TokenizedStory', 'url tokens')


class QuestionContext(
    namedtuple(
        'QuestionContextBase',
        'url context question answer anonymization_info')):

  def ToString(self):
    return '%s\n\n%s\n\n%s\n\n%s\n\n%s' % (
        self.url, self.context, self.question, self.answer,
        '\n'.join(
            [
                key + ':' + value
                for key, value in self.anonymization_info.iteritems()]))


def ReadUrls(filename):
  """Reads a list of URLs.

  Args:
    filename: The filename containing the URLs.

  Returns:
    A list of URLs.
  """

  with open(filename) as f:
    return [line.strip('\n') for line in f]


def ReadMultipleUrls(filename):
  """Reads a list of URL lists.

  Each line in the filename should contain a list of URLs separated by comma.

  Args:
    filename: The filename containing the URLs.

  Returns:
    A list of list of URLs.
  """

  with open(filename) as f:
    return [line.strip('\n').split(',') for line in f]


def WriteUrls(filename, urls):
  """Writes a list of URLs to a file.

  Args:
    filename: The filename to the file where the URLs should be written.
    urls: The list of URLs to write.
  """

  with open(filename, 'w') as f:
    f.writelines(url + '\n' for url in urls)


def Hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string.

  Args:
    s: The string to hash.

  Returns:
    A heximal formatted hash of the input string.
  """

  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def ReadDownloadedUrl(url, corpus):
  """Reads a downloaded URL from disk.

  Args:
    url: The URL to read.
    corpus: The corpus the URL belongs to.

  Returns:
    The content of the URL.
  """

  try:
    with open('%s/downloads/%s.html' % (corpus, Hashhex(url))) as f:
      return f.read()
  except IOError:
    return None


wayback_pattern = re.compile(r'web/([^/]*)/')


def WaybackUrl(urls, max_attempts=6):
  """Retrieves the URL for the latest historic copy using Wayback Machine.

  Args:
    urls: The URL for a specific page (canonical URL + forwarding URL's).
    max_attempts: The maximum attempts at requesting the URL.

  Returns:
    The URL or None if no copy is stored for the URL.

  Raises:
    RuntimeError: Failed to retrieve the URL.
  """

  if not urls:
    return None

  url = urls[0]

  index_collection_url = 'http://archive.org/wayback/available'

  payload = {'url': url}

  attempts = 0

  while attempts < max_attempts:
    try:
      entry_req = requests.get(index_collection_url, params=payload,
                               allow_redirects=False)

      if entry_req.status_code != requests.codes.ok:
        return WaybackUrl(urls[1:], max_attempts)

      entry = entry_req.json()

      if 'closest' not in entry['archived_snapshots']:
        return WaybackUrl(urls[1:], max_attempts)

      wayback_url = entry['archived_snapshots']['closest']['url']
      wayback_url = wayback_pattern.sub(r'web/\g<1>id_/', wayback_url, 1)
      return wayback_url

    except requests.exceptions.ConnectionError:
      pass

    # Exponential back-off.
    time.sleep(math.pow(2, attempts))
    attempts += 1

  raise RuntimeError(
      'Failed to download URL for %s after %d attempts. Please run the script '
      'again.' %
      (url, max_attempts))


def DownloadUrl(url, corpus, max_attempts=5, timeout=5):
  """Downloads a URL.

  Args:
    url: The URL.
    corpus: The corpus of the URL.
    max_attempts: Max attempts for downloading the URL.
    timeout: Connection timeout in seconds for each attempt.

  Returns:
    The HTML at the URL or None if the request failed.
  """

  try:
    with open('%s/downloads/%s.html' % (corpus, Hashhex(url))) as f:
      return f.read()
  except IOError:
    pass

  attempts = 0

  while attempts < max_attempts:
    try:
      req = requests.get(url, allow_redirects=False, timeout=timeout)

      if req.status_code == requests.codes.ok:
        content = req.text.encode(req.encoding)
        with open('%s/downloads/%s.html' % (corpus, Hashhex(url)), 'w') as f:
          f.write(content)
        return content
      elif (req.status_code in [301, 302, 404, 503]
            and attempts == max_attempts - 1):
        return None
    except requests.exceptions.ConnectionError:
      pass
    except requests.exceptions.ContentDecodingError:
      return None
    except requests.exceptions.ChunkedEncodingError:
      return None
    except requests.exceptions.Timeout:
      pass
    except socket.timeout:
      pass

    # Exponential back-off.
    time.sleep(math.pow(2, attempts))
    attempts += 1

  return None


def ParseHtml(story, corpus):
  """Parses the HTML of a news story.

  Args:
    story: The raw Story to be parsed.
    corpus: Either 'cnn' or 'dailymail'.

  Returns:
    A Story containing URL, paragraphs and highlights.
  """

  parser = html.HTMLParser(encoding=chardet.detect(story.html)['encoding'])
  tree = html.document_fromstring(story.html, parser=parser)

  # Elements to delete.
  delete_selectors = {
      'cnn': [
          '//blockquote[contains(@class, "twitter-tweet")]',
          '//blockquote[contains(@class, "instagram-media")]'
      ],
      'dailymail': [
          '//blockquote[contains(@class, "twitter-tweet")]',
          '//blockquote[contains(@class, "instagram-media")]'
      ]
  }

  # Paragraph exclusions: ads, links, bylines, comments
  cnn_exclude = (
      'not(ancestor::*[contains(@class, "metadata")])'
      ' and not(ancestor::*[contains(@class, "pullquote")])'
      ' and not(ancestor::*[contains(@class, "SandboxRoot")])'
      ' and not(ancestor::*[contains(@class, "twitter-tweet")])'
      ' and not(ancestor::div[contains(@class, "cnnStoryElementBox")])'
      ' and not(contains(@class, "cnnTopics"))'
      ' and not(descendant::*[starts-with(text(), "Read:")])'
      ' and not(descendant::*[starts-with(text(), "READ:")])'
      ' and not(descendant::*[starts-with(text(), "Join us at")])'
      ' and not(descendant::*[starts-with(text(), "Join us on")])'
      ' and not(descendant::*[starts-with(text(), "Read CNNOpinion")])'
      ' and not(descendant::*[contains(text(), "@CNNOpinion")])'
      ' and not(descendant-or-self::*[starts-with(text(), "Follow us")])'
      ' and not(descendant::*[starts-with(text(), "MORE:")])'
      ' and not(descendant::*[starts-with(text(), "SPOILER ALERT:")])')

  dm_exclude = (
      'not(ancestor::*[contains(@id,"reader-comments")])'
      ' and not(contains(@class, "byline-plain"))'
      ' and not(contains(@class, "byline-section"))'
      ' and not(contains(@class, "count-number"))'
      ' and not(contains(@class, "count-text"))'
      ' and not(contains(@class, "video-item-title"))'
      ' and not(ancestor::*[contains(@class, "column-content")])'
      ' and not(ancestor::iframe)')

  paragraph_selectors = {
      'cnn': [
          '//div[contains(@class, "cnnContentContainer")]//p[%s]' % cnn_exclude,
          '//div[contains(@class, "l-container")]//p[%s]' % cnn_exclude,
          '//div[contains(@class, "cnn_strycntntlft")]//p[%s]' % cnn_exclude
      ],
      'dailymail': [
          '//div[contains(@class, "article-text")]//p[%s]' % dm_exclude
      ]
  }

  # Highlight exclusions.
  he = (
      'not(contains(@class, "cnnHiliteHeader"))'
      ' and not(descendant::*[starts-with(text(), "Next Article in")])')
  highlight_selectors = {
      'cnn': [
          '//*[contains(@class, "el__storyhighlights__list")]//li[%s]' % he,
          '//*[contains(@class, "cnnStryHghLght")]//li[%s]' % he,
          '//*[@id="cnnHeaderRightCol"]//li[%s]' % he
      ],
      'dailymail': [
          '//h1/following-sibling::ul//li'
      ]
  }

  def ExtractText(selector):
    """Extracts a list of paragraphs given a XPath selector.

    Args:
      selector: A XPath selector to find the paragraphs.

    Returns:
      A list of raw text paragraphs with leading and trailing whitespace.
    """

    xpaths = map(tree.xpath, selector)
    elements = list(chain.from_iterable(xpaths))
    paragraphs = [e.text_content().encode('utf-8') for e in elements]

    # Remove editorial notes, etc.
    if corpus == 'cnn' and len(paragraphs) >= 2 and '(CNN)' in paragraphs[1]:
      paragraphs.pop(0)

    paragraphs = map(str.strip, paragraphs)
    paragraphs = [s for s in paragraphs if s and not str.isspace(s)]

    return paragraphs

  for selector in delete_selectors[corpus]:
    for bad in tree.xpath(selector):
      bad.getparent().remove(bad)

  paragraphs = ExtractText(paragraph_selectors[corpus])
  highlights = ExtractText(highlight_selectors[corpus])

  content = '\n\n'.join(paragraphs)

  return Story(story.url, content, highlights)


def WriteStory(story, corpus):
  """Writes a news story to disk.

  Args:
    story: The news story to write.
    corpus: The corpus the news story belongs to.
  """

  story_string = story.ToString()
  url_hash = Hashhex(story.url)

  with open('%s/stories/%s.story' % (corpus, url_hash), 'w') as f:
    f.write(story_string)


def LoadTokenMapping(filename):
  """Loads a token mapping from the given filename.

  Args:
    filename: The filename containing the token mapping.

  Returns:
    A list of (start, end) where start and
    end (inclusive) are offsets into the content for a token. The list is
    sorted.
  """

  mapping = []

  with open(filename) as f:
    line = f.readline().strip()

    for token_mapping in line.split(';'):
      if not token_mapping:
        continue

      start, length = token_mapping.split(',')

      mapping.append((int(start), int(start) + int(length)))

    mapping.sort(key=lambda x: x[1])  # Sort by start.

  return mapping


def Tokenize(story, corpus):
  """Tokenizes a news story.

  Args:
    story: The Story.
    corpus: The corpus of the news story.

  Returns:
    A TokenizedStory containing the URL and the tokens or None if no token
    mapping was found for the URL.
  """

  s = story.ToString()
  url_hash = Hashhex(story.url)

  mapping_filename = '%s/tokens/%s.txt' % (corpus, url_hash)
  if not os.path.exists(mapping_filename):
    return None

  mapping = LoadTokenMapping(mapping_filename)

  tokens = []

  for (start, end) in mapping:
    tokens.append(s[start:end + 1])

  return TokenizedStory(story.url, tokens)


def LoadEntityMapping(filename):
  """Loads an entity mapping from the given filename.

  Args:
    filename: The filename containing the entity mapping.

  Returns:
    A list of (entity_index, start, end)
    where start and end (inclusive) are token offsets for an entity. The list
    is sorted.
  """

  mapping = []

  with open(filename) as f:
    line = f.readline().strip()

    for entity_mapping in line.split(';'):
      if not entity_mapping:
        continue

      entity_index, start, end = entity_mapping.split(',')

      mapping.append((int(entity_index), int(start), int(end)))

    mapping.sort(key=lambda x: x[2])  # Sort by start.

  return mapping


def Anonymize(tokenized_story, corpus):
  """Anonymizes a tokenized news story.

  Args:
    tokenized_story: A TokenizedStory.
    corpus: The corpus of the tokenized news story.

  Returns:
    A Story containing the URL, anonymized content and anonymized highlights or
    None if no entity mapping exists for the news story.
  """
  url_hash = Hashhex(tokenized_story.url)

  mapping_filename = '%s/entities/%s.txt' % (corpus, url_hash)
  if not os.path.exists(mapping_filename):
    return None

  mapping = LoadEntityMapping(mapping_filename)

  mapping_index = 0
  mapping_len = len(mapping)

  new_tokens = []
  anonymization_info = {}

  i = 0
  while i < len(tokenized_story.tokens):
    if mapping_index < mapping_len and mapping[mapping_index][1] == i:
      entity_index, start, end = mapping[mapping_index]
      anonymized_entity_name = '@entity%d' % entity_index
      new_tokens.append(anonymized_entity_name)
      anonymization_info[anonymized_entity_name] = ' '.join(
          tokenized_story.tokens[start: end + 1]).replace(' - ', '-')

      mapping_index += 1
      i = end + 1
    else:
      new_tokens.append(tokenized_story.tokens[i])

      i += 1

  parts = ' '.join(new_tokens).split(' @ highlight ')

  content = parts[0]
  highlights = parts[1:]

  return AnonymizedStory(
      tokenized_story.url, content, highlights, anonymization_info)

entity_pattern = re.compile(r'@entity\d+')


def GenerateQuestionContexts(anonymized_story, context_token_limit):
  """Generates a list of question/answer pairs given an anonymized news story.

  One question/answer pair is generated for each anonymized entity appearing in
  the question.

  Args:
    anonymized_story: The anonymized news story.
    context_token_limit: If the context of a news story is above the limit, the
        empty list will be returned.

  Returns:
    A list of QuestionContext containing questions and answers.
  """

  result = []

  if anonymized_story.content.count(' ') + 1 > context_token_limit:
    return result

  entities_in_context = set(entity_pattern.findall(anonymized_story.content))

  for highlight in anonymized_story.highlights:
    for match in entity_pattern.finditer(highlight):
      start, end = match.span()

      answer = highlight[start:end]

      if answer not in entities_in_context:
        # Ignore entities that doesn't appear in the content as these will be
        # impossible (or very hard to answer).
        continue

      question = ('%s@placeholder%s' %
                  (highlight[0:start], highlight[end:])).lower()
      context = anonymized_story.content.lower()
      url = anonymized_story.url
      anonymization_info = anonymized_story.anonymization_info
      result.append(
          QuestionContext(url, context, question, answer, anonymization_info))

  return result


def WriteQuestionContext(question_context, corpus, dataset):
  """Writes a question/answer pair to disk.

  Args:
    question_context: The QuestionContext to write containing the question and
        answer.
    corpus: The corpus the question/answer belongs to.
    dataset: One of 'training', 'validation' and 'test'.
  """

  s = question_context.ToString()
  h = Hashhex(s)

  with open('%s/questions/%s/%s.question' % (corpus, dataset, h), 'w') as f:
    f.write(s)


class ProgressBar(object):
  """Simple progress bar.

  Output example:
    100.00% [2152/2152]
  """

  def __init__(self, total=100, stream=sys.stderr):
    self.total = total
    self.stream = stream
    self.last_len = 0
    self.curr = 0

  def Increment(self):
    self.curr += 1
    self.PrintProgress(self.curr)

    if self.curr == self.total:
      print ''

  def PrintProgress(self, value):
    self.stream.write('\b' * self.last_len)
    pct = 100 * self.curr / float(self.total)
    out = '{:.2f}% [{}/{}]'.format(pct, value, self.total)
    self.last_len = len(out)
    self.stream.write(out)
    self.stream.flush()


datasets = ['training', 'validation', 'test']


def UrlMode(corpus, request_parallelism):
  """Finds Wayback Machine URLs and writes them to disk.

  Args:
    corpus: A corpus.
    request_parallelism: The number of concurrent requests.
  """

  for dataset in datasets:
    print 'Finding Wayback Machine URLs for the %s set:' % dataset
    old_urls_filename = '%s/%s_urls.txt' % (corpus, dataset)
    new_urls_filename = '%s/wayback_%s_urls.txt' % (corpus, dataset)

    urls = ReadMultipleUrls(old_urls_filename)

    p = ThreadPool(request_parallelism)
    results = p.imap_unordered(WaybackUrl, urls)

    progress_bar = ProgressBar(len(urls))
    new_urls = []
    for result in results:
      if result:
        new_urls.append(result)

      progress_bar.Increment()

    WriteUrls(new_urls_filename, new_urls)


def DownloadMapper(t):
  """Downloads an URL and checks that metadata is available for the URL.

  Args:
    t: a tuple (url, corpus).

  Returns:
    A pair of URL and content.

  Raises:
    RuntimeError: No metadata available.
  """

  url, corpus = t

  url_hash = Hashhex(url)

  mapping_filename = '%s/entities/%s.txt' % (corpus, url_hash)
  if not os.path.exists(mapping_filename):
    raise RuntimeError('No metadata available for %s.' % url)

  return url, DownloadUrl(url, corpus)


def DownloadMode(corpus, request_parallelism):
  """Downloads the URLs for the specified corpus.

  Args:
    corpus: A corpus.
    request_parallelism: The number of concurrent download requests.
  """

  missing_urls = []
  for dataset in datasets:
    print 'Downloading URLs for the %s set:' % dataset

    urls_filename = '%s/wayback_%s_urls.txt' % (corpus, dataset)
    urls = ReadUrls(urls_filename)

    missing_urls_filename = '%s/missing_urls.txt' % corpus
    if os.path.exists(missing_urls_filename):
      print 'Only downloading missing URLs'
      urls = list(set(urls).intersection(ReadUrls(missing_urls_filename)))

    p = ThreadPool(request_parallelism)
    results = p.imap_unordered(DownloadMapper, izip(urls, repeat(corpus)))

    progress_bar = ProgressBar(len(urls))

    collected_urls = []
    try:
      for url, story_html in results:
        if story_html:
          collected_urls.append(url)

        progress_bar.Increment()
    except KeyboardInterrupt:
      print 'Interrupted by user'

    missing_urls.extend(set(urls) - set(collected_urls))

  WriteUrls('%s/missing_urls.txt' % corpus, missing_urls)

  if missing_urls:
    print ('%d URLs couldn\'t be downloaded, see %s/missing_urls.txt.'
           % (len(missing_urls), corpus))
    print 'Try and run the command again to download the missing URLs.'


def StoreMapper(t):
  """Reads an URL from disk and returns the parsed news story.

  Args:
    t: a tuple (url, corpus).

  Returns:
    A Story containing the parsed news story.
  """

  url, corpus = t

  story_html = ReadDownloadedUrl(url, corpus)

  if not story_html:
    return None

  raw_story = RawStory(url, story_html)

  return ParseHtml(raw_story, corpus)


def StoreMode(corpus):
  for dataset in datasets:
    print 'Storing news stories for the %s set:' % dataset
    urls_filename = '%s/wayback_%s_urls.txt' % (corpus, dataset)
    urls = ReadUrls(urls_filename)

    p = Pool()
    stories = p.imap_unordered(StoreMapper, izip(urls, repeat(corpus)))

    progress_bar = ProgressBar(len(urls))
    for story in stories:
      if story:
        WriteStory(story, corpus)

      progress_bar.Increment()


def GenerateMapper(t):
  """Reads an URL from disk and returns a list of question/answer pairs.

  Args:
    t: a tuple (url, corpus).

  Returns:
    A list of QuestionContext containing a question and an answer.
  """

  url, corpus, context_token_limit = t
  story_html = ReadDownloadedUrl(url, corpus)

  if not story_html:
    return None

  raw_story = RawStory(url, story_html)

  story = ParseHtml(raw_story, corpus)
  tokenized = Tokenize(story, corpus)

  if not tokenized:
    return None

  anonymized = Anonymize(tokenized, corpus)

  if not anonymized:
    return None

  return GenerateQuestionContexts(anonymized, context_token_limit)


def GenerateMode(corpus, context_token_limit):
  for dataset in datasets:
    print 'Generating questions for the %s set:' % dataset

    urls_filename = '%s/wayback_%s_urls.txt' % (corpus, dataset)
    urls = ReadUrls(urls_filename)

    p = Pool()
    question_context_lists = p.imap_unordered(
        GenerateMapper, izip(urls, repeat(corpus), repeat(context_token_limit)))

    progress_bar = ProgressBar(len(urls))
    for question_context_list in question_context_lists:
      if question_context_list:
        for question_context in question_context_list:
          WriteQuestionContext(question_context, corpus, dataset)

      progress_bar.Increment()


def RemoveMode(corpus):
  missing_urls = set(ReadUrls('%s/missing_urls.txt' % corpus))

  for dataset in datasets:
    urls_filename = '%s/wayback_%s_urls.txt' % (corpus, dataset)
    urls = ReadUrls(urls_filename)

    new_urls = []

    for url in urls:
      if url not in missing_urls:
        new_urls.append(url)

    WriteUrls(urls_filename, new_urls)


def main():
  parser = argparse.ArgumentParser(
      description='Generates question/answer pairs')
  parser.add_argument('--corpus', choices=['cnn', 'dailymail'], default='cnn')
  parser.add_argument(
      '--mode', choices=['store', 'generate', 'download', 'urls', 'remove'],
      default='generate')
  parser.add_argument('--request_parallelism', type=int, default=200)
  parser.add_argument('--context_token_limit', type=int, default=2000)
  args = parser.parse_args()

  stories_dir = '%s/stories' % args.corpus
  if not os.path.exists(stories_dir):
    os.mkdir(stories_dir)

  downloads_dir = '%s/downloads' % args.corpus
  if not os.path.exists(downloads_dir):
    os.mkdir(downloads_dir)

  questions_dir = '%s/questions' % args.corpus
  if not os.path.exists(questions_dir):
    os.mkdir(questions_dir)

  for dataset in datasets:
    dataset_dir = '%s/questions/%s' % (args.corpus, dataset)
    if not os.path.exists(dataset_dir):
      os.mkdir(dataset_dir)

  if args.mode == 'store':
    StoreMode(args.corpus)
  elif args.mode == 'generate':
    GenerateMode(args.corpus, args.context_token_limit)
  elif args.mode == 'download':
    DownloadMode(args.corpus, args.request_parallelism)
  elif args.mode == 'urls':
    UrlMode(args.corpus, args.request_parallelism)
  elif args.mode == 'remove':
    RemoveMode(args.corpus)


if __name__ == '__main__':
  main()
