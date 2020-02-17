import os
os.environ["KIVY_NO_CONSOLELOG"] = "1"
from kivy.app import App
from kivy.uix.button import Button


class TestApp(App):

    @staticmethod
    def build():
        return Button(text='Hello World')


def start_window():
    TestApp().run()
