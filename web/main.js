window.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('info-button').addEventListener('click', function() {
        window.pywebview.api.info_button_clicked();
    });

    document.getElementById('play-button').addEventListener('click', function() {
        window.pywebview.api.play_button_clicked();
    });

    document.getElementById('settings-button').addEventListener('click', function() {
        window.pywebview.api.settings_button_clicked();
    });
});