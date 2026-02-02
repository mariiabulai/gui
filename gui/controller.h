#pragma once
#include <QObject>
#include "window.h"
#include "PythonBridge.h"

class Controller : public QObject {
    Q_OBJECT
public:
    Controller(Window* window, QObject* parent = nullptr);

public slots:
    void startReconstruction();

private slots:
    void handleReconstructionResult(bool success, const QString& plyPath, const QString& msg);

private:
    Window* m_window;
    PythonBridge* m_bridge;
};