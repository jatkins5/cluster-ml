#set document(title: "Predicting Galaxy Cluster Merger History from Simulated Radio Emission")
#set page(
  paper: "us-letter",
  margin: 1in,
  numbering: "1",
  number-align: center,
)
#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.65em, first-line-indent: 1.5em)

#set heading(numbering: "1.")
#show heading.where(level: 1): it => {
  v(0.8em)
  text(size: 12pt, weight: "bold", it)
  v(0.2em)
}
#show heading.where(level: 2): it => {
  v(0.4em)
  text(size: 11pt, weight: "bold", style: "italic", it)
}

// Header block
#align(center)[
  #text(size: 14pt, weight: "bold")[
    Predicting Galaxy Cluster Merger History from Simulated Radio and X-ray Emission
  ]

  #v(0.4em)
  #text(size: 11pt)[Jeremy Atkins]
  #h(1em)
  #text(size: 10pt)[Advisor: Ian Dell'Antonio]
  #h(1em)
  #text(size: 10pt)[#datetime.today().display("[month repr:long] [day], [year]")]
]

#v(1em)

#include "sections/01_introduction.typ"
#include "sections/02_survey_coverage.typ"
#include "sections/03_simulation_data.typ"
#include "sections/04_models.typ"
#include "sections/05_results.typ"
#include "sections/06_discussion.typ"
#include "sections/07_conclusion.typ"

#bibliography("references.bib", style: "american-physics-society")
