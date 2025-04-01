import os

def execute_command(command):
    try:
        stream = os.popen(command)
        output = stream.read()
        print("Output:\n", output)
    except Exception as e:
        print("Error:\n", str(e))

execute_command("llamafactory-cli env")
execute_command("llamafactory-cli version")