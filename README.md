# cspy (Q-Scope)

A tool for scanning interactions with a tangible grid of acrylic plates, based on the MIT's [CityScoPy](https://cityscope.media.mit.edu/docs/TUI/CityScoPy). It is used in the [Q-Scope Project](q-scope.readthedocs.io) of the [Department of Resilient Energy Systems](www.uni-bremen.de/res).


## Adjustments:

- enhanced using parts of [HCU CSL Code]()
- reads from tagged acrylic plates instead of black and white LEGO
- can read values of optical sliders
- offers some more tools to tweak the image (like added grayscale layer) and thus improves detection accuracy


## Recommended folder structure

```
project qScope
└───cspy
│   └───modified MIT CityScoPy Token Tag Decoder
└───data
|   └───outputs
|      └───output_[timestamp]
|         (simulation-specific output)
|         └───buildings_clusters_[timestamp].csv
|         (exportierte Gebäudeliste von Frontend)
|         └───simulation_parameters_[timestamp].xml
|         (xml-Datei mit allen Simulationsparametern zum Starten des headless modes)
|         └───connections
|         |       Export der Anschlussquoten
|         └───emissions
|         |      gebäudespezifische und aggregierte Quartiersemissionen
|         └───snapshot
|               von GAMA produzierte Grafiken
└───q100_abm
│   │   GAMA workspace folder
│   └───q100
│       │   Trend Model
│    	└───models
|       │    └───qscope_ABM.gaml
|       └───__data__ symlink zu data-Ordner (unten))
└───qScope_infoscreen
│       infoscreen (NodeJS/ JavaScript)
└───qScope_frontend
        projection (Python)

```

where:
- cspy: https://github.com/quarree100/cspy
- data: has to be linked from Seafile server
- GAMA: https://github.com/quarree100/q100_abm
- qScope_infoscreen: https://github.com/quarree100/qScope_infoscreen
- qScope_frontend: https://github.com/quarree100/qScope_frontend
