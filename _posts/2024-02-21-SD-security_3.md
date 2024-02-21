---
title: Cybersecurity3 - Introduction to secure systems - University of Lille
author:
  name: Pierre Lague & François Muller (@franzele21)
  link: 
date: 2024-02-21 09:45:00 +0800
categories: [Studies, U-Lille]
tags: [Python, Security, C]
math: true
mermaid: true
image:
  src: '/assets/posts/security/header.jpg'
  width: 800
  height: 600
---

# Cybersecurity Class🛡️

## Overview
This class is proposed as part of the first year of masters curriculum at the University of Lille and it overlooked by Pr Gilles GRIMAUD. It delves into four primary themes: permissions and secure file systems, SQL injections, XSS attacks and prevention methods, and cryptography with RSA and elliptic curves.


## Foreword:

At first glance, this system presents a problem because it depends on two individuals, significantly increasing its risk of compromise. Additionally, it is not possible to make it truly infallible with university resources alone. However, we will endeavor to make it as easy to use as possible without making it easily compromised.

## Question 1:

> Let's propose a solution so that decryption is only made possible by two-factor authentication from both responsible parties.

- Context:
    - The server is located in a secure area with guarded access and identity proof (identity card and verification in a registry of authorized personnel). The responsible parties are rigorously searched.

    - Server activation:
        - Before entering, the responsible parties verify that there is no one in the server room.
        - The server can be powered on using two consoles, each for one of the two responsible parties (the two consoles cannot see each other and are located on either side of the server so that neither of the responsible parties can see each other's information).
        - Upon arriving at the activation stations, the responsible parties must have their USB key around their necks to be able to insert it quickly and simultaneously (within a 3-second margin).
        - The two responsible parties insert their keys at the same time (within a 3-second margin). Then they have 20 seconds and 1 attempt to enter their respective passwords correctly. Otherwise, the protocol is canceled, and guards are alerted. **The system verifies that there are two entries at the two activation stations to prevent one responsible party from performing the procedure alone with the other's information**.
        - Once the keys and passwords are inserted, the elements are as follows:
            - $p_1$, $k_1$: the password/key pair of responsible party $1$.
            - $p_2$, $k_2$: the password/key pair of responsible party $2$.

            - We transform them as follows:
                - $k_1$ mod($p_2$) * $k_2$ mod($p_1$) = $H$. The modulo allows us to use the pairs $k_1$,$p_1$ and $k_2$,$p_2$ on either of the activation stations.
            - Then, the server verifies that $H$ is equal to a pre-stored value from which extraction of the pairs $k_1$,$p_1$ and $k_2$,$p_2$ is **impossible**.
            
            **If this is the case, the server is activated; otherwise, the operation is canceled, and an alert is triggered.**
        - Upon leaving the server room, the responsible parties will be (re)searched.

Assuming that the activation has been successful, we can take the risk of using $H$ to decrypt the file containing the names and card information based on an equality check.

To encrypt our data, we use the $PGP$ protocol (Pretty Good Privacy: https://en.wikipedia.org/wiki/Pretty_Good_Privacy). This protocol seemed to us to be the right one to use due to its notoriety and ease of use via the command line. It is a military-grade and fast protocol (Schneier, Bruce (October 9, 1995). Applied Cryptography page 587). For each operation (consultation, addition, deletion) on the bank data file, it is decrypted, editable (via available operations) for the duration of the operation in a temporary file, then at the end of the operation, the bank data file is encrypted again, and the temporary file is deleted.

Once the activation is done for the first time, subsequent uses of this 2FA authentication principle will lead directly to the decryption of the file containing the names and card information. The file is encrypted with a combination of the active users' keys present in the `possible_hash` file.

## Q2:

### Service $1.i$ 

For obvious reasons, lacking bodyguards, we will not implement searches but the internal parts of the server from the insertion of keys and then passwords. In the folder `SECURE SYSTEM`, you will find all the codes associated with the implementation of our security system.

- The file `server.py` manages client authentication.
- The file `client1.py` manages the sending of clients' keys and passwords to the server. Once this is done, commands are executed directly on a command prompt linked to the server.
- The file `functions.py` contains all the functions associated with the actions that the responsible parties can perform on the server.
- The file `launch_server.sh` starts the server and the client. This is the system's entry point.
- The file `pgp_encryption.py` implements the encryption and decryption functionalities of the bank data file.

Illustration of system usage:

The command
```bash
./launch_server.py
```
Produces:
```bash
127.0.0.1 - - [09/Feb/2024 14:23:57] "POST /key HTTP/1.1" 200 -
127.0.0.1 - - [09/Feb/2024 14:23:58] "POST /password HTTP/1.1" 200 -
127.0.0.1 - - [09/Feb/2024 14:24:07] "POST /key HTTP/1.1" 200 -
[+] User key input took too long... aborting
```
If the user takes too long to enter their key or password.
For obvious reasons, in a real system, the specification of the reason for cancellation would not be specified. Here it is for the clarity of the reviewer.

If the responsible parties enter the wrong password or key, the result is as follows:
```bash
 * Debugger PIN: 932-979-265
127.0.0.1 - - [09/Feb/2024 14:27:56] "POST /key HTTP/1.1" 200 -
127.0.0.1 - - [09/Feb/2024 14:27:56] "POST /password HTTP/1.1" 200 -
127.0.0.1 - - [09/Feb/2024 14:27:57] "POST /key HTTP/1.1" 200 -
[-] NUHUUU DURR you're not getting in...
```

Otherwise, if the responsible parties enter their information on time:
```bash
127.0.0.1 - - [09/Feb/2024 14:25:58] "POST /key HTTP/1.1" 200 -
127.0.0.1 - - [09/Feb/2024 14:25:58] "POST /password HTTP/1.1" 200 -
127.0.0.1 - - [09/Feb/2024 14:25:59] "POST /key HTTP/1.1" 200 -
[+] Double authentication passed...
[+] System Online...
[+] Awaiting instructions : 
         [1] Insert information 
         [2] Delete information 
         [3] Lookup information

 
         [4] Read file 
         [5] Exit system
[*] Select option : 
```
Then the user can choose to execute one of the 5 options:

>Option [1]
```bash
127.0.0.1 - - [09/Feb/2024 14:29:00] "POST /key HTTP/1.1" 200 -
127.0.0.1 - - [09/Feb/2024 14:29:01] "POST /password HTTP/1.1" 200 -
127.0.0.1 - - [09/Feb/2024 14:29:02] "POST /key HTTP/1.1" 200 -
[+] Double authentication passed...
[+] System Online...
[+] Awaiting instructions : 
         [1] Insert information 
         [2] Delete information 
         [3] Lookup information 
         [4] Read file 
         [5] Exit system
[*] Select option : 1
gpg: AES256.CFB encrypted data
gpg: encrypted with 1 passphrase
[+] Inserting information in the secret file...
[*] Name : Lague
[*] Card Number : 25042001
[+] Data Inserted...
```

>Option [3]

```bash
[+] Awaiting instructions : 
         [1] Insert information 
         [2] Delete information 
         [3] Lookup information 
         [4] Read file 
         [5] Exit system
[*] Select option : 3
[*] Enter name : Lague
gpg: AES256.CFB encrypted data
gpg: encrypted with 1 passphrase
Lague / 25042001
```

>Option [2]

```bash
[+] Awaiting instructions : 
         [1] Insert information 
         [2] Delete information 
         [3] Lookup information 
         [4] Read file 
         [5] Exit system
[*] Select option : 2
[*] Enter name : Lague
[*] Enter card number : 25042001
gpg: AES256.CFB encrypted data
gpg: encrypted with 1 passphrase
[+] Information deleted...
```

>Option [4]

```bash
[+] Awaiting instructions : 
         [1] Insert information 
         [2] Delete information 
         [3] Lookup information 
         [4] Read file 
         [5] Exit system
[*] Select option : 4
gpg: AES256.CFB encrypted data
gpg: encrypted with 1 passphrase
Muller / 26042001
Muller / 26789423
```

>Option [5]
```bash
[*] Select option : 5
pierre.lague.etu@115p10:~/Documents/ISI/isi-tp3-cryptography/SECURE SYSTEM$ 
```

>The contents of the bank data file:
```
�
	�p��+<��X������s�Ph
```
## Q3:

In order to differentiate between the responsible parties and their legal representatives, the representatives should have a key and password that allows them to access the system. It is possible to store hashes of authorized persons in a system file `possible_hash`. None of these hashes would allow someone to trace back to the passwords or keys.

At the program level, it will be sufficient to just store the hash versions of the responsible parties along with the legal representatives.

## Q4:

So, we have added to the `possible_hash` file the new hash of the representatives so that they can log in and decrypt the bank data files.

## Q5:

Repudiation is the action of denying access/action performed by someone who has access to a service or who has had access to that service. On the other hand, non-repudiation consists of ensuring that an action on the data performed on behalf of a user (after authentication) cannot be repudiated by the user. i.e., the user cannot deny having performed the actions on the system.

To implement this system, it is necessary to ensure that there is a hierarchy between the responsible parties and their legal representatives. i.e., a responsible party can delete their legal representative, but a legal representative cannot delete their responsible party. In this way, it is necessary to be able to identify users in the `possible_hash` file to delete the correct one based on the key stored as a hash.

As a result, the possible hashes multiply. For each new user, 2 new hashes are needed.

- 2 responsible parties -> 1 hash
- 2 responsible parties + 1 representative -> 3 hashes
- 2 responsible parties + 2 representatives -> 6 hashes

In order to manage all possible user combinations.

However, our system is based on a single key per user pair (combination of their credentials). This makes implementing repudiation difficult.

## Q6:

The structure of `possible_hash` takes the following form: `hash,key1,key2`.

```python
with open("possible_hash") as file:
    all_password = file.read().split("\n")
all_password = [x.split(",") for x in all_password]
for i, password in enumerate(all_password):
    all_password[i] = int(password[0])


keys = [int(input("Enter the key from the deleted person: "))]
                high_hierarchy = input("Does this person have a legal representant [y/n]: ")
                if high_hierarchy in ["y", "Y", "Yes", "yes", "YES"]:
                    keys.append(int(input("Enter the key of the person: ")))
                delete_user(keys)

def delete_user(keys):
    with open("possible_hash", "r") as file:
        content = file.read()
        content = content.split("\n")
        content = [x.split(",") for x in content]
        content = [[int(x) for x in y] for y in content]
    new_content = []
    for line in content:
        if line[1] not in keys and line[2] not in keys:
            new_content.append(line)
    new_content = [[str(x) for x in y] for y in new_content]
    new_content = [",".join(x) for x in new_content]
    new_content = "\n".join(new_content)

    with open("possible_hash", "w") as file:
        file.write(new_content)
    print("[+] User deleted!")
```

>Execution:

The server asks for the key to delete. Then if this person has a legal representative:
- if so, the responsible party and their representative are deleted
- if not, it means it is a representative, so only that user whose key was entered is deleted.

```bash
[*] Select option: 5
Enter the key from the deleted person: 5
Does this person have a legal representant [y/n]: n
[+] User deleted!
[+] Awaiting instructions:
     [1] Insert information
     [2] Delete information
     [3] Lookup information
     [4] Read file
     [5] Delete user 
     [6] Exit
     
[*] Select option:
```

Similarly, it is possible to add a user to the list of possible hashes.

```bash
127.0.0.1 - - [15/Feb/2024 10:44:15] "POST /key HTTP/1.1" 200 -
127.0.0.1 - - [15/Feb/2024 10:44:16] "POST /password HTTP/1.1" 200 -
127.0.0.1 - - [15/Feb/2024 10:44:16] "POST /key HTTP/1.1" 200 -
[+] Double authentication passed...
[+] System Online...
[+] Awaiting instructions:
     [1] Insert information
     [2] Delete information
     [3] Lookup information
     [4] Read file
     [5] Delete user 
     [6] Add User
     [7] Exit
     
[*] Select option: 6
[+] New user's key: 6
[+] New user's password: 9
[+] Key of who you legally represent: 1
[+] Enter the key of your homologue: 3
[*] Request accepted: possible duo
[+] Homologue enter your password: 4
[+] Add another user duo?[y/n]n
[+] Awaiting instructions:
...

```

If we log in with the key/password pairs 6-9 and 3-4, we pass the double authentication.

---

🔒 Secure your systems, shield your data 🔒