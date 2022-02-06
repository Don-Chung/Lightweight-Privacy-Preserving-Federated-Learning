def judge_prime(number):
    status = 0
    if number < 1 or type(number) != int:
        return 'type error'
    else:
        for i in range(2, number):
            if number % i == 0:
                status = 1
        return status


def isPrime(a, b):
    while b != 0:
        temp = b
        b = a % b
        a = temp
    if a == 1:
        return 1
    else:
        return 0


def find_prine(number):
    primeList = []
    for i in range(1, number + 1):
        ans = isPrime(i, number)
        if ans == 1:
            primeList.append(i)
    byg = []
    prime_temp = []
    for j in primeList:
        for i in range(1, len(primeList) + 1):
            prime_temp.append(j ** i % number)
        prime_temp.sort()
        if primeList == prime_temp:
            byg.append(j)
        else:
            pass
        prime_temp = []
    return byg


def calculation(number, p, g):
    return pow(g, number) % p
