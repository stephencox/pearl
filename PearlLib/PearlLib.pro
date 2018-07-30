TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS += PearlLib PearlLibConsole PearlLibTest
PearlLibConsole.depends = PearlLib
PearlLibTest.depends = PearlLib
