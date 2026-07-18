import numpy as np
from utils import read_file, normalize, calculate_price


def main():
    theta_0 = 0
    theta_1 = 0

    with open('values.txt', 'r') as f:
        theta_0 = f.readline()
        theta_1 = f.readline()

    greeting = "Welcome to your car price evaluator system.\
        \n\tPlease enter the milage traveled by the car: "
    milage = input(f"\x1B[3m\x1B[32m{greeting}\x1B[0m")

    if int(milage) < 0:
        raise ValueError("\x1B[30m\x1B[41mValue Error: ")

    df = read_file('data.csv')
    km = np.array(df['km'])
    price = calculate_price(theta_0=float(theta_0), theta_1=float(theta_1),
                            milage=normalize(km, milage=int(milage)))
    print(f"\x1B[36mYour car price is estimated to: \x1B[1m{price:.2f}$")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\x1B[1m\x1B[31mFile Error: {e}\x1B[0m\
              \n\x1B[1m\x1B[31mPlease check if the file "
              "'values.txt' exists\x1B[0m")
    except ValueError as e:
        print(f"\x1B[30m\x1B[41m{e}: Milage must be a postive number.\x1B[0m")
    except KeyboardInterrupt as e:
        print(f"\x1B[1m\x1B[31m\nProgram unexpectedly stopped. {e}\x1B[0m")
