#pragma once
#include <QWidget>
#include <QMainWindow>
#include <QListWidgetItem>
#include "Viewer3DWidget.h"

namespace Ui { class window; }

class Window : public QMainWindow { // Используем QMainWindow как в .ui
    Q_OBJECT
public:
    explicit Window(QWidget *parent = nullptr);
    ~Window();

    // Методы для контроллера
    QStringList getSelectedImages() const; // Теперь возвращаем список файлов
    void updateStatus(const QString &msg, int progress = -1);
    void show3DModel(const QString &plyPath);
    void setControlsEnabled(bool enable);
    
    // Метод, чтобы спросить пользователя файл калибровки (так как его нет в UI)
    QString promptForCalibrationFile();

signals:
    void startRequested();

private slots:
    void on_addPhotosButton_clicked();
    void on_removePhotoButton_clicked();
    void on_startButton_clicked();

private:
    Ui::window *ui;
    Viewer3DWidget *m_viewer;
};