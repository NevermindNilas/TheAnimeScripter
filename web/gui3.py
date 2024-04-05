import webview

class Api:
    def __init__(self, window):
        self.window = window

    @webview.window.api
    def info_button_clicked(self):
        print("Info button clicked")

    @webview.window.api
    def play_button_clicked(self):
        print("Play button clicked")

    @webview.window.api
    def settings_button_clicked(self):
        print("Settings button clicked")


def main():
    window = webview.create_window(
        "The Anime Scripter",
        "web/main.html",
        vibrancy=True,
        resizable=False,
        width=1366,
        height=768,
    )
    api = Api(window)
    window.js_api = api
    webview.start()


if __name__ == "__main__":
    main()
