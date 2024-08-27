from streamlit.web import cli as stcli
import sys

def main():
    sys.argv = ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
