"""
Different typographic attack styles.

Each function returns parameters that
describe how the malicious prompt should
be rendered on the document.
"""

import random


def footer_style():

    return {

        "position": "footer",

        "font_size": 18,

        "color": (0, 0, 0),

        "rotation": 0
    }


def tiny_style():

    return {

        "position": "footer",

        "font_size": 8,

        "color": (0, 0, 0),

        "rotation": 0
    }


def gray_style():

    return {

        "position": "footer",

        "font_size": 10,

        "color": (160, 160, 160),

        "rotation": 0
    }


def margin_style():

    return {

        "position": "left_margin",

        "font_size": 10,

        "color": (120, 120, 120),

        "rotation": 90
    }


def rotated_style():

    return {

        "position": "footer",

        "font_size": 10,

        "color": (0, 0, 0),

        "rotation": 15
    }


STYLES = [

    footer_style,

    tiny_style,

    gray_style,

    margin_style,

    rotated_style

]