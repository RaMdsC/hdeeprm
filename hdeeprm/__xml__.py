"""
Helper module for XML type hinting.
"""

from defusedxml.lxml import _etree as exml

XMLElement = exml.ElementBase.mro()[1]
