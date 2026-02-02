#pragma once
#include <QObject>
#include <QStringList>

class controller : public QObject {
    Q_OBJECT
public:
    explicit controller(QObject *parent = nullptr);

    // Метод для старту
    void startReconstruction(const QStringList& images);

signals:
    // Сигнали для GUI
    void reconstructionStarted();
    void reconstructionFinished();
};