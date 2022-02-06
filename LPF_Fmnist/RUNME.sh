STRING1="If you want to modify the parameters, please modify it in file para.ini, including the number of users, rounds of the protocol, whether to run in static or dynamic mode etc."
echo $STRING1
STRING2="Start creating users..."
echo $STRING2
eval python Create_Users.py
STRING3="Start Running..."
echo $STRING3
eval python Execute.py