TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS += PearlLib PearlLibConsole
PearlLibConsole.depends = PearlLib
