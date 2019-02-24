"""
Helper module for XML type hinting.

Attributes:
    XMLElement (lxml.etree._Element): XML Element for type hinting.
"""

from defusedxml.lxml import _etree as exml

XMLElement = exml.ElementBase.mro()[1]
