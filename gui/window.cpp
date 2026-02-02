#include "window.h"
#include "ui_window.h"
#include "controller.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QShortcut> // Додаємо для клавіші Delete

window::window(controller* controller, QWidget *parent) : QMainWindow(parent), 
    ui(new Ui::window), m_controller(controller) {
    
    ui->setupUi(this);

    // --- 1. Кнопка Add Photos ---
    connect(ui->addPhotosButton, &QPushButton::clicked, this, [this](){
        QStringList files = QFileDialog::getOpenFileNames(
            this, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        );
        // Додаємо, не видаляючи старі (щоб можна було дозавантажити)
        ui->imageListWidget->addItems(files);
    });

    // --- 2. НОВА КНОПКА Remove Selected ---
    // Функція видалення винесена в лямбду, щоб використовувати її і для кнопки, і для клавіші
    auto deleteSelectedPhotos = [this]() {
        // Отримуємо список вибраних елементів
        QList<QListWidgetItem*> items = ui->imageListWidget->selectedItems();
        
        // Якщо нічого nie вибрано - нічого не робимо
        if (items.isEmpty()) return;

        // Видаляємо кожен вибраний елемент
        for (auto item : items) {
            // Важливо: delete видаляє об'єкт з пам'яті
            delete ui->imageListWidget->takeItem(ui->imageListWidget->row(item));
        }
    };

    connect(ui->removePhotoButton, &QPushButton::clicked, this, deleteSelectedPhotos);

    // --- 3. Гаряча клавіша DELETE ---
    // Створюємо шорткат: коли фокус на вікні і тиснеш Del -> видаляє фото
    QShortcut* shortcut = new QShortcut(QKeySequence::Delete, ui->imageListWidget);
    connect(shortcut, &QShortcut::activated, this, deleteSelectedPhotos);


    // --- 4. Кнопка Start ---
    connect(ui->startButton, &QPushButton::clicked, this, [this](){
        
        QStringList currentFiles;
        for(int i = 0; i < ui->imageListWidget->count(); ++i) {
            currentFiles.append(ui->imageListWidget->item(i)->text());
        }

        if (currentFiles.size() < 2) {
            QMessageBox::warning(this, "Error", "Upload at least 2 images!");
            return;
        }

        m_controller->startReconstruction(currentFiles);
    });

    // --- 5. Сигнали контролера ---
    connect(m_controller, &controller::reconstructionStarted, this, [this](){
        ui->startButton->setEnabled(false);
        ui->startButton->setText("Processing...");
        ui->addPhotosButton->setEnabled(false);
        ui->removePhotoButton->setEnabled(false); // Блокуємо видалення під час роботи
        ui->progressBar->setRange(0, 0); 
    });

    connect(m_controller, &controller::reconstructionFinished, this, [this](){
        ui->startButton->setEnabled(true);
        ui->startButton->setText("Start Reconstruction");
        ui->addPhotosButton->setEnabled(true);
        ui->removePhotoButton->setEnabled(true); // Розблокуємо
        ui->progressBar->setRange(0, 100);
        ui->progressBar->setValue(100);
        
        QMessageBox::information(this, "Success", "Reconstruction completed!");
    });
}

window::~window() {
    delete ui;
}