TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.c

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../PearlLib/release/ -lPearlLib0
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../PearlLib/debug/ -lPearlLib0
else:unix: LIBS += -L$$OUT_PWD/../PearlLib/ -lPearlLib

INCLUDEPATH += $$PWD/../PearlLib
DEPENDPATH += $$PWD/../PearlLib

INCLUDEPATH += ../../external/json-c

linux: {
    QMAKE_CFLAGS_RELEASE += -march=native
}
