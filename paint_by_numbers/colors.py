import numpy as np
from dataclasses import dataclass

"""
https://www.amazon.de/-/en/Marabu-1210000000201-Water-Based-Suitable-Waterproof/dp/B012Z5TW8S/ref=sr_1_5?keywords=acrylfarben&qid=1636134894&qsid=260-6400767-2825044&sr=8-5&sres=B012Z5TW8S%2CB0881WTJPV%2CB085VWMS8B%2CB08GMC3SZ9%2CB07R3G853M%2CB08PYJDBVB%2CB00KA8BVUA%2CB00OQPIEVM%2CB08GM9NSMR%2CB08B37HZCD%2CB0882ZLJML%2CB00PWKF6DE%2CB07YGTXZSR%2CB08R1RFXJ2%2CB078CYZJ2C%2CB08XW11XV4%2CB07PNPWC91%2CB08PFKB8HG%2CB08DHLLQNC%2CB0091Y8YOU&srpt=PAINT&th=1
https://www.marabu-creative.com/de/produkte/kuenstlerfarben/marabu-acrylfarben-18er-sortierung-basic-18-x-36-ml-tuben-1210000000201/
"""


MARABU_COLORS_DICT = dict(
    TITANWEISS=[237, 238, 240],
    ZITRONENGELB=[244, 207, 57],
    GELBOCKER=[178, 117, 50],
    ORANGEGELB=[255, 86, 21],
    ZINNOBERROT=[254, 59, 27],
    SCHARLACHROT=[215, 33, 29],
    PURPURROT=[178, 25, 30],
    SMARAGDGRUEN=[53, 165, 83],
    VIRIDIANGRUEN=[0, 117, 85],
    SAFTGRUEN=[49, 89, 37],
    HIMMELBLAU=[39, 132, 191],
    CYANBLAU_DUNKEL=[3, 77, 150],
    PHTHALOCYANINBLAU=[19, 17, 82],
    SIENNA=[164, 96, 47],
    SIENNA_GEBRANNT=[179, 73, 34],
    UMBRA_GEBRANNT=[101, 57, 44],
    VANDYKE_BRAUN=[38, 37, 42],
    SCHWARZ=[20, 12, 25]
)

MARABU_COLORS = np.array(list(MARABU_COLORS_DICT.values()))


ARTIST_AND_CO_COLORS_LIST = [[105,57,47],
                        [54,39,36],
                        [105,193,142],
                        [42,67,63],
                        [117,188,218],
                        [35,35,71],
                        [31,40,57],
                        [8,8,8],
                        [255, 236, 207],
                        [255,216,124],
                        [113, 54, 38],
                        [168, 64, 37],
                        [236, 38, 51],
                        [229, 32, 52],
                        [155, 40, 37],
                        [206, 36, 65]]

ARTIST_AND_CO_COLORS = np.array(ARTIST_AND_CO_COLORS_LIST)

