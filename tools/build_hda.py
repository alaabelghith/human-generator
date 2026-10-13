"""Refresh the Human Generator HDA (houdini/otls/human_generator.hda) from houdini/python/human_generator/parms.py.

Run inside Houdini (Python Shell), with the package installed:
    exec(open(hou.getenv("HUMAN_GENERATOR") + "/tools/build_hda.py").read())
The .hda in the repository is the shipped file; this script rewrites its parameter interface and its
PythonCook code, so edit parms.py, run this, and commit the .hda.
"""
import importlib
import os
import sys

import hou

root = hou.getenv("HUMAN_GENERATOR")
sys.path.insert(0, os.path.join(root, "houdini", "python"))
import human_generator.parms as P  # noqa: E402
importlib.reload(P)

path = os.path.join(root, "houdini", "otls", "human_generator.hda").replace("\\", "/")
defs = [d for d in hou.hda.definitionsInFile(path) if d.nodeTypeName() == "alaa::human_generator"]
if not defs:
    raise RuntimeError("alaa::human_generator not found in %s" % path)
d = defs[0]
d.setParmTemplateGroup(P.parms())
d.addSection("PythonCook", P.COOK)
d.setMinNumInputs(0)
d.setMaxNumInputs(1)
d.setIcon("SOP_python")
d.setComment("Human Generator %s: real-unit bodies (SOMA / Anny) + text-to-motion (NVIDIA Kimodo)." % __import__("human_generator").__version__)
d.setVersion(__import__("human_generator").__version__)
d.save(path)
print("Saved", path)
