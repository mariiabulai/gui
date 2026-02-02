#pragma once
#include <QObject>
#include <QStringList>
#include <QFileInfo>
#include <QTimer> // Для імітації затримки

class PythonBridge : public QObject
{
    Q_OBJECT
public:
    explicit PythonBridge(QObject *parent = nullptr) : QObject(parent) {}

    // Метод, який викликає AppController
    void runReconstruction(const QStringList& imagePaths);

signals:
    // Сигнал, який AppController очікує отримати
    void backendTaskFinished(bool success, const QFileInfo& plyPath, const QString& errorMessage);

private slots:
    // Слот, який викликається QTimer
    void mockReconstructionComplete();

private:
    QString m_mockPlyPath = "output/model_output.ply";
};