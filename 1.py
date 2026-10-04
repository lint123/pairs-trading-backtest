game_is_running = True

while game_is_running:

    print("\n--- Welcome to Doxford Academy ---")
    choice = input("Type 'quit' to exit or 'hello' to continue: ")

    if choice == "quit":
        print("Goodbye!")
        game_is_running = False

    else:
        choice != "quit"
        print("Yay!")
        game_is_running = True
