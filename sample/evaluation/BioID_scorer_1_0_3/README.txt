Copyright 2017 The MITRE Corporation. All rights reserved.

MITRE public release information: Approved for Public Release;
Distribution Unlimited. Case Number 17-2967.

BIOID SCORER
------------

This directory contains the scorer for the BioID evaluation. Examples
of how to invoke the scorer are below.

The scorer requires Python 2.7, with the lxml package available. All
other required packages are included. This scorer has not been tested
on Windows, but there's no reason it shouldn't work.

The included packages in lib/python are:

- munkres-1.0.5.4
- PyBioC-1.0.2

The copyright for these packages can be found in these
directories. The copyright notice for all other files is:

"Copyright 2017 The MITRE Corporation. All rights reserved."

USAGE
-----

% python bioid_score.py
Usage: python bioid_score.py [ --verbose <n> ] [ --force ] [ --test_equivalence_classes <json> ] [ --type_restriction <type>(,<type>...) ] out_dir reference_dir (<run_name>:)run_dir [ (<run_name>:)run_dir ... ]

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
  listed below in SCORING METHODS AND CRITERIA.
  
EXAMPLE
-------

Given a reference directory of files /data/bioid_train_references, and
a candidate run in /data/bioid_run_system1, the following invocation
will provide reasonable progress pacifiers (via --verbose), clear the
output directory if it's present (via --force) and deposit the scores
in /data/bioid_scores, with an identifier "system1" for the candidate
run in the spreadsheets:

% python bioid_score.py --verbose 1 --force \
/Users/ningshixian/Desktop/BC6_Track1/BioID_scorer_1_0_3/data/bioid_scores /Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/caption_bioc system1:/Users/ningshixian/Desktop/BC6_Track1/BioIDtest/caption_bioc_unannotated 

INPUTS
------

The reference corpus, as well as each corpus to be evaluated, must be
in BioC format. This is the format that the training and test corpora
have been distributed in. For more information on BioC, visit this Web
site:

http://bioc.sourceforge.net/

The PyBioC library which this scorer uses is referenced there:

https://github.com/2mh/PyBioC/

NOTE: the offsets which the BioID scorer expects for mentions are BYTE
offsets, as required in the BioC DTD.

SCORING METHODS AND CRITERIA
----------------------------

Each annotation is assigned to a label, according to the prefix of the
value of its "type" infon. The "type" infon is expected either to be
an ID in the relevant ontology (the normalized case), or a prefixed
string indicating the type (the unnormalized case). The prefixes are
assigned to labels as follows:

Uniprot: (normalized), protein: (unnormalized), NCBI gene: (normalized), gene: (unnormalized) -> gene_or_protein
CHEBI: (normalized), PubChem: (normalized), molecule: (unnormalized) -> small_molecule
GO: (normalized), subcellular: (unnormalized) -> cellular_component
CL: (normalized), CVCL_ (normalized), cell: (unnormalized) -> cell_type_or_line
Uberon: (normalized), tissue: (unnormalized) -> tissue_or_organ
NCBI taxon: (normalized), organism: (unnormalized) -> organism_or_species

These prefixes are case sensitive. Annotations with no "type" infon
value, or with a "type" infon value which doesn't match one of the
prefixes listed, will be ignored. Please consult the reference
training corpus for examples of each "type" infon value. The reference
"type" info values may contain alternative normalizations, separated
with a vertical bar ("|"). The hypothesis submissions must not contain
alternatives.

While the BioC representation supports discontinuous annotations,
there are no such annotations in the training corpus, and the BioID
scorer ignores the multiple spans; the initial endpoint is the start
index of the <location> with the lowest offset, and the final endpoint
is the offset + length of the <location> with the highest offset +
length.

The scorer will report scores at three levels: the individual caption
level, the document level, and the corpus level. At the last two
levels, scores will be aggregated PER LABEL from the previous
level. The scorer performs no aggregation across labels, since
different participants might choose to produce annotations for only a
subset of the possible labels.

The scorer will report scores in 4 conditions: ( all annotations
vs. normalized annotations only ) X ( strict span match vs. span overlap
). For each condition, mention-level recall/precision/fmeasure will be
reported.

The scorer will also compute recall/precision/fmeasure on the
normalized IDs which are found, both micro-averaged and
macro-averaged. This procedure is complicated by the presence of
alternatives in the reference, which makes the computation of missing
elements more challenging. We describe the procedure here.

(There are actually two types of alternatives. In addition to the
alternatives explicitly represented for certain mentions, there is a
global set of equivalence classes (see
resources/equivalence_classes.json) which, e.g., specify
correspondences between Entrez and Uniprot. This latter set of
equivalence classes is applied throughout this procedure, and does not
affect the description of the procedure, and will be ignored for the
moment.)

For each label:

(1) First, we construct three sets from the reference normalizations:
the singleton reference normalizations, which have no alternatives; the
alternative reference normalizations, each of which is a set of the
alternatives found in a single normalization; and the union of the
singleton normalizations and all the sets in the alternative normalizations,
which is the full reference normalizations.

(2) We compute the match set as the intersection between the
hypothesis normalizations and the full reference normalizations.

(3) We compute the spurious set as the hypothesis normalizations minus the
match set.

(4) We compute the initial missing set as the singleton reference
normalizations minus the match set.

(5) We remove from the set of alternative reference normalizations any set
which intersects with either the match set or the initial missing
set. These reference normalizations have already been satisfied.

(6) We are left with all the alternative reference normalizations for
which no member was either found or is known to be missing already. We
compute the "missing bump" as the lesser of the size of the remaining
alternative reference normalizations or the size of the union of the sets
in the alternative reference normalizations. We add this "bump" to the
size of the initial missing set to get the missing count.

At this point, we compute precision, recall and f-measure in the
normal way. These scores are reported for each of the 4 score
conditions, although they will be identical for each.

Note that this normalization evaluation is completely independent of the
normalization of individual mentions. In other words, if a caption has a
mention A with normalization ID1, and a mention B with normalization
ID2, and both are assigned the same label, a submission which contains
mention A with normalization ID2 and mention B with normalization ID1 will be
judged completely correct. In a subsequent version of this scorer, we
may add a mention-sensitive normalization metric.

OUTPUT
------

The output directory will contain five files:

caption_scores.csv: these are the scores per individual caption
document_scores.csv: these are the scores per document
corpus_scores.csv: these are the overall corpus scores

pair_details_normalized_only.csv: these are the details for the
  individual annotation pairs, for the case where only normalized
  annotations are considered
pair_details_no_norm_restriction.csv: these are the details for
  the individual annotation pairs, for the case where all annotations
  are considered

The first three files contain the following columns:

run: the run name (either provided on the command line or "run<n>")
norm_condition: "normalized" or "any"
span_condition: "strict" (strict extent match) or "overlap"
document: the PMC ID of the document (caption and document only)
figure: the name of the figure (caption only)
caption_count: how many captions contributed to this row
label: the label, as described above
match: the number of matching annotations
refclash: the number of annotations with this label in the 
  reference which clash with the hypothesis annotation they're paired
  with either in label or span
missing: the number of annotations with this label in the 
  reference which have no pair in the hypothesis
refonly: refclash + missing
reftotal: match + refonly
hypclash: the number of annotations with this label in the hypothesis
  which clash with the reference annotation they're paired with
  either in label or span
spurious: the number of annotations with this label in the hypothesis
  which have no pair in the reference
hyponly: hypclash + spurious
hyptotal: match + hyponly
precision: match / hyptotal
recall: match / reftotal
fmeasure: 2 * (( precision * recall) / (precision + recall))
norm_match: the number of normalizations (or equivalence classes)
  that occur both in the reference annotations in the caption and
  the hypothesis annotations in the caption
norm_missing: the number of normalizations (or equivalence classes)
  that occur only in the reference annotations in the caption
norm_spurious: the number of normalizations (or equivalence
  classes) that occur only in the hypothesis annotations in the caption
norm_precision_micro: norm_match / (norm_spurious + norm_match)
norm_recall_micro: norm_match / ( norm_missing + norm_match)
norm_fmeasure_micro: 2 * (( norm_precision_micro * norm_recall_micro) / (norm_precision_micro + norm_recall_micro))
norm_precision_macro: average of all norm_precision_micro
  precisions for the individual captions
norm_recall_macro: average of all norm_recall_micro
  precisions for the individual captions
norm_fmeasure_macro: average of all norm_fmeasure_micro
  precisions for the individual captions

The last two files contain the following columns:

run: the run name (either provided on the command line or "run<n>")
document: the PMC ID of the document (caption and document only)
figure: the name of the figure (caption only)
status: the status of the pair. Possible values are match, missing,
  spurious, spanclash, labelclash (the last two can cooccur). In the
  overlap span condition, spanclash without labelclash counts as a
  match. 
reflabel: the label of the reference annotation, as described above
refstart: the start byte offset of the reference annotation
refend: the end byte offset of the reference annotation
reftext: the text covered by the reference annotation
reftype: the value of the "type" infon for the reference annotation
reftype_eqclass: the equivalence class for the reference annotation
hyplabel: the label of the hypothesis annotation, as described above
hypstart: the start byte offset of the hypothesis annotation
hypend: the end byte offset of the hypothesis annotation
hyptext: the text covered by the hypothesis annotation
hyptype: the value of the "type" infon for the hypothesis annotation
hyptype_eqclass: the equivalence class for the hypothesis annotation

If the status value is "missing", all the hypothesis values will be
blank in that row; if the status value is "spurious", all the
reference values will be blank in that row.
