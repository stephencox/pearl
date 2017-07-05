TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.c

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../PearlLib/release/ -lPearlLib
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../PearlLib/debug/ -lPearlLib
else:unix: LIBS += -L$$OUT_PWD/../PearlLib/ -lPearlLib

INCLUDEPATH += $$PWD/../PearlLib
DEPENDPATH += $$PWD/../PearlLib
