#ifndef PYTHONBRIDGE_H
#define PYTHONBRIDGE_H

#include <QObject>
#include <QProcess>
#include <QString>

class PythonBridge : public QObject {
    Q_OBJECT
public:
    explicit PythonBridge(QObject *parent = nullptr);

    // Метод для запуска процесса
    void runReconstruction(const QString &imagesPath, const QString &calibrationFile);

signals:
    // Сигналы для обновления интерфейса
    void progressUpdated(const QString &msg);
    void reconstructionFinished(bool success, const QString &plyPath, const QString &msg);

private slots:
    // Слоты для обработки ответов от Python
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onReadyReadStandardOutput();
    void onReadyReadStandardError();

private:
    QProcess *m_process;
    QString m_lastError; // <--- ВОТ ЭТОЙ СТРОЧКИ НЕ ХВАТАЛО
};

#endif // PYTHONBRIDGE_H