#!/usr/bin/env python3
"""Fix malformed XML files in a Doxygen XML output directory.

Doxygen occasionally emits broken XML (mismatched tags) for complex C++
constructs.  These files crash Breathe's parser.  This script validates
every .xml file with the standard library's ElementTree parser and replaces
any that fail to parse with a minimal valid stub, printing a warning for
each.

Usage:
    python clean_doxygen_xml.py <xml_directory>
"""

import sys
import os
import xml.etree.ElementTree as ET

# Minimal valid Doxygen compound XML that Breathe can parse without error.
_STUB = """\
<?xml version='1.0' encoding='UTF-8'?>
<doxygen xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:noNamespaceSchemaLocation="compound.xsd" version="">
  <compounddef id="{compound_id}" kind="class" language="C++">
    <compoundname>{compound_name}</compoundname>
    <briefdescription>
      <para>(XML was malformed and has been replaced by a stub.)</para>
    </briefdescription>
    <detaileddescription/>
    <location file="unknown"/>
  </compounddef>
</doxygen>
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <xml_directory>", file=sys.stderr)
        return 1

    xml_dir = sys.argv[1]
    if not os.path.isdir(xml_dir):
        print(f"Not a directory: {xml_dir}", file=sys.stderr)
        return 1

    fixed = 0
    for fname in sorted(os.listdir(xml_dir)):
        if not fname.endswith(".xml"):
            continue
        fpath = os.path.join(xml_dir, fname)
        try:
            ET.parse(fpath)
        except ET.ParseError as exc:
            # Derive a compound id and name from the filename.
            # e.g. "classDNDS_1_1Foo.xml" → id="classDNDS_1_1Foo", name="DNDS::Foo"
            base = fname.removesuffix(".xml")
            name = base.replace("class", "").replace("struct", "").replace("_1_1", "::")
            stub = _STUB.format(compound_id=base, compound_name=name)
            with open(fpath, "w") as f:
                f.write(stub)
            print(f"  replaced malformed XML: {fname}  ({exc})")
            fixed += 1

    if fixed:
        print(f"  fixed {fixed} malformed XML file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
