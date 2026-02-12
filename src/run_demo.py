import sys
import os
from streamlit.web import cli as stcli



def main():
    # دیگر GPU را غیرفعال نمی‌کنیم
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" <-- حذف شد

    # فقط لاگ‌های اضافی را حذف می‌کنیم
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    print("🚀 Starting Streamlit App (Auto Device Selection)...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, 'app.py')

    sys.argv = ["streamlit", "run", app_path]

    sys.exit(stcli.main())


if __name__ == '__main__':
    main()

