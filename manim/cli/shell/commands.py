import cmd
import os
import readline

import click
import cloup


class ManimShell(cmd.Cmd):
    intro = "Welcome to the Manim Shell!\nType 'help' for more info.\nType 'q', 'quit' or 'exit' to close the shell, or press Ctrl + C.\n"
    prompt = "[manim] "
    directory = os.path.dirname(os.path.realpath(__file__))
    history_filename = f"{directory}/command_history"

    def __init__(self, ctx):
        cmd.Cmd.__init__(self)
        self.ctx = ctx.parent
        self.main = ctx.parent.command

    def cmdloop(self):
        try:
            super().cmdloop()
        except KeyboardInterrupt:
            print()
            self.postloop()
            exit(1)

    def preloop(self):
        if os.path.exists(self.history_filename):
            readline.read_history_file(self.history_filename)

    def postloop(self):
        readline.set_history_length(1000)
        curr_length = readline.get_current_history_length()
        command = readline.get_history_item(curr_length)
        if command in ("q", "quit", "exit"):
            readline.remove_history_item(curr_length - 1)
        readline.write_history_file(self.history_filename)

    def do_help(self, line):
        if line == "":
            print(self.main.get_help(self.ctx))
        else:
            params = line.split()
            subcommand, args = params[0], params[1:]
            subcommand = self.main.commands.get(subcommand)
            if subcommand:
                print(subcommand.get_help(self.ctx))
            else:
                print("*** Unknown command. Use 'help' for a list of commands.")

    def default(self, line):
        params = line.split()
        subcommand, args = params[0], params[1:]
        if subcommand in ("q", "quit", "exit"):
            return True

        subcommand = self.main.commands.get(subcommand)
        if not subcommand:
            subcommand = self.main.commands.get("render")
            args = params

        subcommand.parse_args(self.ctx, args)
        self.ctx.forward(subcommand)


@cloup.command()
@click.pass_context
def shell(ctx):
    ManimShell(ctx).cmdloop()
