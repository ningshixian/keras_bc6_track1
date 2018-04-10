# Copyright 2017 The MITRE Corporation. All rights reserved.
# Approved for Public Release; Distribution Unlimited. Case Number 17-2967.

import os, sys, getopt, shutil

SCORER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCORER_ROOT, "lib", "python"))
sys.path.insert(0, os.path.join(SCORER_ROOT, "lib", "python", "PyBioC-1.0.2", "src"))
sys.path.insert(0, os.path.join(SCORER_ROOT, "lib", "python", "munkres-1.0.5.4"))

import bioid_scorer

LEGAL_TYPES = ["cell_type_or_line", "cellular_component", "gene_or_protein",
               "organism_or_species", "small_molecule", "tissue_or_organ"]

def Usage():
    print >> sys.stderr, """Usage: python bioid_score.py [ --verbose <n> ] [ --force ] [ --test_equivalence_classes <json> ] [ --type_restriction <type>(,<type>...) ] out_dir reference_dir (<run_name>:)run_dir [ (<run_name>:)run_dir ... ]

Python 2.7 required. Your Python environment must include lxml, which is required by PyBioC.

out_dir: the output directory in which the score files will be
  placed. Will be created if it doesn't exist. If it does exist, the
  scorer will exit unless the --force option is provided.
reference_dir: a directory containing a reference corpus. This corpus
  should consist of XML files in BioC format, one per document. The
  name of the file is the PMC ID of the document from which the
  captions within it were extracted.
(<run_name>:)run_dir: a directory representing a system run,
  consisting of XML files in BioC format. This argument may be
  repeated (so the scorer can compare multiple runs to the same reference
  simultaneously). You may provide an optional prefix which
  will be the name of the run in the output spreadsheets; all unnamed
  runs will be numbered in the order in which they're presented on the
  command line.

--verbose: an integer indicating verbosity. Default is 0, which is
  almost completely silent. A value of 1 produces reasonable pacifiers
  as progress indicators.
--force: if the output directory exists, the scorer will fail to
  overwrite it, and exit. If this option is provided, the directory
  will be removed and regenerated.
--test_equivalence_classes: a file of equivalence classes in the same format as
  resources/equivalence_classes.json, which represents the equivalence classes
  found in the test data (not included in the code archive for obvious reasons)
--type_restriction: a comma-separated list of labels to restrict the output to 
  (for those participants who only consider particular labels). Legal labels are
  %s""" % (", ".join(LEGAL_TYPES))
    sys.exit(1)

if (sys.version_info.major != 2) or (sys.version_info.minor != 7):
    print >> sys.stderr, "Python 2.7 is required. Exiting."
    sys.exit(1)

try:
    import lxml
except ImportError:
    print("Failed to import the lxml package, which is required by the BioID scorer. Exiting.")
    sys.exit(1)

ops, args = getopt.getopt(sys.argv[1:], "", ["verbose=", "force", "test_equivalence_classes=", "type_restriction="])
VERBOSE = 0
FORCE = False
TEST_EQUIVALENCE_CLASSES = None
TYPE_RESTRICTIONS = None

for k, v in ops:
    if k == "--verbose":
        try:
            VERBOSE = int(v)
        except ValueError:
            print >> sys.stderr, "--verbose must be an integer. Exiting."
            sys.exit(1)
    elif k == "--force":
        FORCE = True
    elif k == "--test_equivalence_classes":
        TEST_EQUIVALENCE_CLASSES = os.path.abspath(v)
    elif k == "--type_restriction":
        v = [s.strip() for s in v.split(",")]
        for subv in v:
            if subv not in LEGAL_TYPES:
                print >> sys.stderr, "%s is not a legal type for type restrictions. Exiting." % subv
                sys.exit(1)
        TYPE_RESTRICTIONS = v
    else:
        Usage()

if len(args) < 3:
    Usage()

# OK, all preliminaries have been satisfied.

[OUT_DIR, GOLD_DIR] = [os.path.abspath(x) for x in args[:2]]

if os.path.exists(OUT_DIR):
    if not os.path.isdir(OUT_DIR):
        print >> sys.stderr, "out_dir exists, but is not a directory. Exiting."
        sys.exit(1)
    elif not FORCE:
        print >> sys.stderr, "out_dir already exists. To forcibly remove and regenerate it, use --force. Exiting."
        sys.exit(1)
    else:
        shutil.rmtree(OUT_DIR)
    
RUN_DIRS = []
for runDir in args[2:]:
    toks = runDir.split(":", 1)
    if len(toks) > 1:
        RUN_DIRS.append((toks[0], os.path.abspath(toks[1])))
    else:
        RUN_DIRS.append((None, os.path.abspath(runDir)))

os.makedirs(OUT_DIR)
bioid_scorer.score.Score(GOLD_DIR, RUN_DIRS, OUT_DIR, verbose = VERBOSE, testEquivalenceClasses = TEST_EQUIVALENCE_CLASSES, typeRestrictions = TYPE_RESTRICTIONS)
