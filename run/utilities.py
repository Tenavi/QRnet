import argparse

def yn_input(message):
    while True:
        try:
            user_input = input(message + ' Enter (y/n):\n').lower()
            if user_input == 'y':
                return True
            elif user_input == 'n':
                return False
            else:
                raise ValueError
        except ValueError:
            print(user_input, "is not a valid input. Must enter 'y' or 'n'...")

def make_parser(
        rs=False, dt=False, mt=False, a=False, xd=False, o=False, w=False,
        v=False, io=False
    ):
    parser = argparse.ArgumentParser()

    if rs:
        parser.add_argument(
            '-rs', '--seed', dest='random_seed', type=int,
            default=None, help='random seed'
        )
    if dt:
        parser.add_argument(
            '-dt', '--data_type', dest='data_type', type=str,
            default='train', help='data type to generate. Options are "train" or "test"'
        )
    if mt:
        parser.add_argument(
            '-mt', '--timestamp', dest='timestamp', type=int,
            default=None, help='10 digit model timestamp'
        )
    if a:
        parser.add_argument('-a', '--architecture', dest='architecture',
            type=str, default=argparse.SUPPRESS,
            help='NN architecture to use'
        )
    if xd:
        parser.add_argument(
            '-xd', '--distance', dest='X0_distance', type=float,
            default=None, help='initial condition norm'
        )
    if o:
        parser.add_argument(
            '-o', '--solve_open_loop', dest='solve_open_loop',
            action='store_true',
            help='solve the open loop OCP for each initial condition?'
        )
    if v:
        parser.add_argument(
            '-v', '--verbose', dest='verbose', type=int,
            default=0, help='how much information to print'
        )
    if io:
        parser.add_argument(
            '-io', '--import_open_loop', dest='import_open_loop',
            action='store_true',
            help='import open loop OCP solutions from test data'
        )

    return parser
