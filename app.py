"""Source and frozen application entry point."""
import logging
import sys
from pathlib import Path


def main():
    if '--project-self-test' in sys.argv:
        from comic_color.project_smoke import run
        if not run(Path(sys.argv[sys.argv.index('--project-self-test') + 1])):
            raise SystemExit(1)
        return
    if '--self-test' in sys.argv:
        from comic_color.smoke import run
        if not run(Path(sys.argv[sys.argv.index('--self-test') + 1])):
            raise SystemExit(1)
        return
    from comic_color.config import data_dir
    from logging.handlers import RotatingFileHandler
    logging.basicConfig(handlers=[RotatingFileHandler(data_dir() / 'application.log', maxBytes=2000000, backupCount=2, encoding='utf-8')], level=logging.INFO)
    from comic_color.gui import ColoringApp
    ColoringApp().mainloop()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception('Application failed')
        if '--self-test' in sys.argv or '--project-self-test' in sys.argv:
            raise
        import tkinter.messagebox
        tkinter.messagebox.showerror('漫画上色工作台', '启动失败，请查看本机应用数据目录中的 application.log。')
        raise
