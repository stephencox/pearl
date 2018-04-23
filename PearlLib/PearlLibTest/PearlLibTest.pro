QT       += testlib
QT       -= gui
TARGET = tst_pearllibtesttest
CONFIG   += console
CONFIG   -= app_bundle
TEMPLATE = app
DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += tst_pearllibtesttest.cpp

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../PearlLib/release/ -lPearlLib0
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../PearlLib/debug/ -lPearlLib0
else:unix: LIBS += -L$$OUT_PWD/../PearlLib/ -lPearlLib

INCLUDEPATH += $$PWD/../PearlLib
DEPENDPATH += $$PWD/../PearlLib

linux: {
    QMAKE_CFLAGS_RELEASE += -march=native
}
