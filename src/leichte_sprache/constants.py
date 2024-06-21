# # # # # Data Management # # # # #

### database ###
CRAWLER_TABLE = "crawled_texts"
DATASET_SINGULAR_TABLE = "dataset_singular"
DATASET_TRANSLATED_TABLE = "dataset_singular_translated"

## dataset labels ###
LS_LABEL = "leichte_sprache"
SG_LABEL = "standard_german"

### column names ###
LS_COLUMN = "leichte_sprache"
SG_COLUMN = "standard_german"
SRC_COLUMN = "source"
TEXT_COLUMN = "text"
URL_COLUMN = "url"
CRAWL_TS_COLUMN = "crawl_timestamp"
TITLE_COLUMN = "title"
RELEASE_COLUMN = "release_date"
FULL_TEXT_COLUMN = "full_text"
ID_COLUMN = "id"
TRANSLATED_COLUMN = "translated"
CHAT_COLUMN = "chat"
PROMPTS_COLUMN = "prompts"
ORIG_IDS_COLUMN = "orig_ids"

### Crawler arguments ###
DLF_DICT = "dlf_dict"
DLF_NEWS = "dlf_news"
NDR = "ndr"
MDR_DICT = "mdr_dict"
MDR_NEWS = "mdr_news"
HURRAKI = "hurraki"
PARLAMENT = "das_parlament"
ALL_SOURCES = "all_sources"

### PDF sections DAS PARLAMENT ###
ARTICLE_TITLE = "article_title"
ARTICLE_SUBTITLE = "article_subtitle"
TITLE = "title"
LOGO = "logo"
SUBTITLE = "subtitle"
ARTICLE_TEXT = "article_text"


# # # # # Prompts and example data # # # # #

### prompts ###
LS_SYSTEM_PROMPT_DICT = {
    "role": "system",
    "content": "Leichte Sprache hat besondere Regeln. Sätze müssen sehr kurz und verständlich sein. Jeder Satz enthält nur eine Aussage. Es werden nur Aktivsätze verwendet. Sätze bestehen aus den Gliedern Subjekt-Verb-Objekt, z. B. Das Kind streichelt den Hund. Es wird immer das gleiche Wort für die gleiche Sache benutzt. Verneinungen werden, wenn möglich, positiv umformuliert, z. B. 'Das kostet nichts.' zu 'Das ist umsonst'. Der Konjunktiv wird vermieden. Der Genitiv wird durch Fügungen mit 'von' ersetzt, z. B. 'Das Haus des Lehrers' durch 'Das Haus vom Lehrer'. Schwierige Wörter werden erklärt. Zusammengesetzte Wörter werden getrennt, zum Beispiel wird 'Weltall' zu 'Welt-All'. Du bist Übersetzer von Standarddeutsch in Leichte Sprache.",
}
LS_USER_PROMPT_TEXT = "Schreibe den folgenden Text nach den Regeln der Leichten Sprache. Text:\n{text_user}\nText in Leichter Sprache:"
