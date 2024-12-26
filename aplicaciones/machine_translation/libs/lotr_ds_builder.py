import requests
import re

# https://github.com/howa003/complete-elvish-lotr-subtitles
subtitles_lotr_fellowship = "https://raw.githubusercontent.com/howa003/complete-elvish-lotr-subtitles/refs/heads/main/1_Fellowship-of-the-Ring/release-034811/ForeignOnly.The.Lord.of.the.Rings.The.Fellowship.of.the.Ring.2001.EXTENDED.MkvCage.ZoowlCZ.srt"
subtitles_lotr_2tower = "https://raw.githubusercontent.com/howa003/complete-elvish-lotr-subtitles/refs/heads/main/2_The-Two-Towers/release-035525/ForeignOnly.The.Lord.of.the.Rings.The.Two.Towers.2002.EXTENDED.1080p.BrRip.x264.YIFY.ZoowlCZ.srt"
subtitles_lotr_king = "https://raw.githubusercontent.com/howa003/complete-elvish-lotr-subtitles/refs/heads/main/3_Return-of-the-King/release-042310/ForeignOnly.The.Lord.of.the.Rings.The.Return.of.the.King.EXTENDED.2003.720p.BrRip.x264.BOKUTOX.YIFY.ZoowlCZ.srt"

OPEN_BRACKET = "["
CLOSE_BRACKET = "]"
FRAGMENT_REGEX = r"\d*\n\s*\d+:\d+:\d+,\d+\s*-->\s*\d+:\d+:\d+,\d+"
MUSIC_TOKEN = "♫"
CLEANING_LINE_TOKEN = r'\]\s*'+MUSIC_TOKEN+'?\s*'
OPEN_BRACKET = "["
CLOSE_BRACKET = "]"


def download_file(url):
  response = requests.get(url)
  srt_text = response.text
  return srt_text

def contains_brackets(line):
  return OPEN_BRACKET in line and CLOSE_BRACKET in line

def lines_are_valid(lines):
   return len(lines) == 2 and contains_brackets(lines[0]) and contains_brackets(lines[1])

def extract_lang(raw_line):
  line_fragment= raw_line.split(OPEN_BRACKET)[1]
  line_lang = line_fragment.split(CLOSE_BRACKET)[0]
  return line_lang.strip()

def clean_line(raw_line):
  try:
    clean_line = re.split(CLEANING_LINE_TOKEN,raw_line)[1]
    clean_line = clean_line.replace(MUSIC_TOKEN,"")
    clean_line = re.sub(r"</?i>","",clean_line)
    clean_line = re.sub(r"\s*\.\.\.\s*","",clean_line)
    return clean_line.strip()
  except:
    print("ERROR CLEANING LINE: ",raw_line)

def add_trainig_lines(lotr_dict, foreign_lang, clean_foreign_line, clean_en_line):
  if foreign_lang not in lotr_dict.keys():
    lotr_dict[foreign_lang] = { "translation": [] }
  lotr_dict[foreign_lang]['translation'] += [{ "en" : clean_en_line, "fo" : clean_foreign_line }]

def build_dataset(url, lotr_dict):
  # Download and create fragments
  srt_text = download_file(url)
  fragments = re.split(FRAGMENT_REGEX, srt_text)
  # Itearte over fragments and split into valid pair of lang lines
  for fragment in fragments:
    fragment = fragment.strip()
    lines = fragment.splitlines()
    if lines_are_valid(lines):
      foreign_line = lines[0]
      en_line = lines[1]
      # Clean lines
      foreign_lang = extract_lang(foreign_line)
      clean_foreign_line = clean_line(foreign_line)
      clean_en_line = clean_line(en_line)
      add_trainig_lines(lotr_dict, foreign_lang, clean_foreign_line, clean_en_line)

def show_dataset_stats(dict_dataset):
  dict_stats = {}
  for lang in dict_dataset.keys():
    size = len(dict_dataset[lang]['translation'])
    dict_stats[lang] = size
  print(dict_stats)


def build_datasets():
  lotr_dict_training = {}
  lotr_dict_validating = {}
  # Notar que en la pelicula de las 2 torres se habla mucho Sindarin
  # esta configuracion crea unos datsets mas balanceados
  build_dataset(subtitles_lotr_fellowship, lotr_dict_training)
  build_dataset(subtitles_lotr_king, lotr_dict_training)
  build_dataset(subtitles_lotr_2tower, lotr_dict_validating)
  build_dataset(subtitles_lotr_2tower, lotr_dict_training)

  show_dataset_stats(lotr_dict_training)
  show_dataset_stats(lotr_dict_validating)
  return lotr_dict_training, lotr_dict_validating