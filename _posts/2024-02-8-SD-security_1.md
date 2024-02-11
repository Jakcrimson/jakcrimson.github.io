---
title: Cybersecurity1 - Introduction to secure systems - University of Lille
author:
  name: Pierre Lague & François Muller (@franzele21)
  link: 
date: 2024-02-08 09:45:00 +0800
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

## Themes Explored

## Permissions and Secure File Systems
Understanding the importance of proper permissions and secure file systems in maintaining data integrity and confidentiality.

---

## Question 1

We cannot write to the file because we do not have write permissions for the current user (-r--rw-r--). Thus, when attempting to save the write buffer of the file, we get a "Permission Denied" message.

## Question 2

Changing permission '-x' for a directory means removing the ability for a non-sudoer user who is not part of the power group to enter this directory.
By running the following operations:

```bash
>su ubuntu
>cd ~ 
>mkdir mydir
>chmod g-x mydir # removing execution rights from the directory
>ls -al # notice that the group x is gone
>cd mydir # can enter the directory because for the current user, there is still the x right.
```

When using the user `toto`, it is not possible to enter the `mydir` directory. Group members will receive this message:

```bash
bash: cd: mydir: Permission denied
```

## Question 3

Our script is as follows:
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int main() {
    // Displaying user and group identifiers
    printf("EUID: %d\n", geteuid());
    printf("EGID: %d\n", getegid());
    printf("RUID: %d\n", getuid());
    printf("RGID: %d\n", getgid());

    // Reading the file mydir/mydata.txt
    const char *filename = "mydir/mydata.txt";
    int fd = open(filename, O_RDONLY);

    if (fd == -1) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    /*
    Code associated with setting up the read buffer
    ... (see suid.c)
    */

    // Reading the file content
    ssize_t bytesRead = read(fd, buffer, st.st_size);
    if (bytesRead == -1) {
        perror("Error reading file");
        free(buffer);
        close(fd);
        exit(EXIT_FAILURE);
    }

    // Displaying the file content
    printf("\nContent of file %s:\n", filename);
    write(STDOUT_FILENO, buffer, bytesRead);

    // Freeing resources
    free(buffer);
    close(fd);

    return 0;
}
```

- When executing the script with the ubuntu user, the following output is obtained:

```
ubuntu@vm1:~$ ./a.out 
EUID: 1000
EGID: 1000
RUID: 1000
RGID: 1000

Content of file mydir/mydata.txt:
Hello World
``` 

Because the file is in read permission for the ubuntu user and ubuntu group.

- When executing the script with the user toto, the following output is obtained:

```
toto@vm1:/home/ubuntu$ ./a.out 
EUID: 1001
EGID: 1001
RUID: 1001
RGID: 1001

Content of file mydir/mydata.txt:
Hello World
```

We notice that the process IDs are different and that the file content can still be read.
`mydir` and `mydata` are in read and execute permission for the `ubuntu` group.

When using the `set-user-id` flag, it is possible for a user in the `ubuntu` group to read the content of the `mydata.txt` file even if it is not in execute mode for groups.

```bash
chmod u+s mydata.txt  
ls -l
#output
-rwxrwxr-x 1 ubuntu ubuntu 16568 Jan 10 15:40 a.out
-rwSrw-r-- 1 ubuntu ubuntu    18 Jan 10 16:10 mydata.txt
-rw-rw-r-- 1 ubuntu ubuntu  1507 Jan 10 15:38 print_ids.c

root@vm1:/home/ubuntu# su toto
toto@vm1:/home/ubuntu$ ./a.out 
EUID: 1001
EGID: 1001
RUID: 1001
RGID: 1001

Content of file mydir/mydata.txt:
Hello World !!!!!
```

We find the processes of the `toto` user.

## Question 4

Our script is as follows:

```python
#!/usr/bin/env python3
import os

def print_user_group_ids():
    # Displaying effective user and group identifiers
    print(f"EUID: {os.geteuid()}")
    print(f"EGID: {os.getegid()}")

if __name__ == "__main__":
    # Executing the function to display user and group identifiers
    print_user_group_ids()
```
We assign `set-user-id` to the python file, and the following permissions: `-rwsrw-r-- 1 ubuntu ubuntu   519 Jan 10 16:33 suuid.py`.

```bash
toto@vm1:/home/ubuntu$ ls -l 
total 28
-rwxrwxr-x 1 ubuntu ubuntu 16568 Jan 10 15:40 a.out
drwxrwxr-x 2 ubuntu ubuntu  4096 Jan 10 16:10 mydir
-rwsrw-r-- 1 ubuntu ubuntu   519 Jan 10 16:33 suuid.py
toto@vm1:/home/ubuntu$ python3 suuid.py 
EUID: 1001
EGID: 1001
```
Thanks to `set-user-id`, the `toto` user is able to execute the script. We find the `toto` user IDs.

The set-user-id (setuid) flag is a security mechanism in Unix/Linux operating systems. It allows a user to execute a program with privileges higher than those it normally has.

The usefulness of the setuid flag is often associated with situations where a program requires elevated privileges to perform certain operations, but the normal user should not have these privileges permanently.

Regarding the modification of attributes without asking the administrator, this can be achieved if the administrator specifically provided for it in the system configuration. For example, there could be a program with setuid that allows the user to modify some of their own information in the `/etc/passwd` file without requiring full administrative privileges. 

For example, if we try to modify the `suid` in a python script executed by `ubuntu` without admin rights, we get the following output:

```bash
ubuntu@vm1:~$ python3 suuid.py 
EUID: 1000
EGID: 1000
UIDs: (1000, 1000, 1000)
UIDs: (1000, 1000, 1000) #modification after displaying the ids
Traceback (most recent call last):
  File "/home/ubuntu/suuid.py", line 22, in <module>
    print_user_group_ids()
  File "/home/ubuntu/suuid.py", line 13, in print_user_group_ids
    os.setresuid(ruid, euid, suid

) 
PermissionError: [Errno 1] Operation not permitted
```

```python
#the code in question:
ruid = 1001
euid = 1000
suid = 1001
os.setresuid(ruid, euid, suid) 
print(f"UIDs: {os.getresuid()}")
print(f"UIDs: {os.getresgid()}")
```


## Question 5

We obtain the following result:
```bash
toto:x:1001:1001::/home/toto:/bin/bash
```
in the form `Name:Password:UserID:PrincipleGroup:Gecos: HomeDirectory:Shell`

```bash
>ls -al /usr/bin/chfn
-rwsr-xr-x 1 root root 72712 Nov 24  2022 /usr/bin/chfn
```
We notice that the `set-user-id` is enabled, which means that users without superuser rights can execute the `chfn` command to modify their information.

---

Naturally, `chfn` executed with root allows to change information about a user (real name, username, information, etc.).

If we try to run the `chfn` command with the `toto` user, the user must enter a password, presumably, that of the user (in this case `root`). Once entered, we can modify the `toto` user's information.

```bash
toto@vm1:/home/ubuntu$ chfn
Password: 
Changing the user information for toto
Enter the new value, or press ENTER for the default
	Full Name: 
	Room Number []: 45
	Work Phone []: 06123485678
	Home Phone []: 031234678

```

When we check the contents of `etc/passwd`, we notice that the information has been updated:

```bash
toto:x:1001:1001:,45,06123485678,031234678:/home/toto:/bin/bash
```

## Question 6

The `/etc/shadow` file contains passwords and is used to increase the level of password security by limiting access to hashed password data to users with superuser rights.

## Question 7

For setting up the structure, we decide to create an additional group that contains `group_a` and `group_b` to avoid having to use Access Control Lists.

Among others, we used the following functions:

- `groupadd`: to create a group
- `adduser`: to create users
- `chgrp`: to assign groups to a user
- `chown`: to change the owner of an entity (file, directory)
- `chmod`: to modify permissions associated with a file/directory
    - use of the `sticky-bit +t` to limit actions within a directory.

At the end of the setup, here are the permissions for the shared tree:

```bash
lambda_a@vm1:/root/partage$ ls -l
total 20
-rwxr-xr-x 1 root  root         945 Jan 18 13:52 admin.sh
drwxrwx--T 4 admin groupe_a    4096 Jan 18 13:55 dir_a
drwxrws--T 3 admin groupe_b    4096 Jan 18 13:52 dir_b
drwsr-s--- 2 admin groupe_gene 4096 Jan 18 13:52 dir_c
-rwxr-xr-x 1 root  root         548 Jan 18 09:49 lambda_a.sh
```

Place the bash scripts in the *question7* directory.

## Question 8

Our programs for questions 8 and 9 are combined. We take into account password creation, updating the passwd file, and encrypting passwords in passwd with `id:passwd`.

The executed scripts give the following results:

>With the admin user when they want to delete a file in `dir_a`.
```bash
admin@vm1:/root/partage$ ./rmg_test.sh
File to delete: dir_a/suppme
Your file exists!
Enter your password :
SYSTEM : Your userid and password have been added to the file 
Access granted
Deleting the file
```

>With the lambda_a user when they want to delete a file in `dir_b`.
```bash
lambda_a@vm1:/root/partage$ ./rmg_test.sh
File to delete: dir_a/suppme
Your file exists!
Enter your password :
SYSTEM : Your userid and password have been added to the file 
Access denied
```



## Question 9

see q8. and directory q9.

>User is already in the database

```bash
display before

15764:12Zrt0LAKAbjk
Enter your password :
SYSTEM : You are already registered in the passwd file
display after

15764:12Zrt0LAKAbjk
# user_id:crypt(password)
```

>User added to the database

```bash
display before

Enter your password :
SYSTEM : Your userid and password have been added to the file 
display after

15764:12tJAuWgu.oe6
# user_id:crypt(password)
```

## Question 10

see directory q10. 

an example of execution:

>here the server is listening

```bash

admin@vm1:/root/partage$ ./test-server.sh 
SERVER
Socket successfully created..
Socket successfully binded..
Server listening..
server accept the client...
```

>when launching the test client script, the client connects to the server and can communicate.
We didn't have time to link the scripts from questions 8 and 9 properly, nor to implement the fork() properly.

```bash
admin@vm1:/root/partage$ ./test-client.sh 
CLIENT
Socket successfully created..
connected to the server..
Enter your username: pier
Enter your password: oui
From Server : You are connected!

Enter the string : list ./dir_a #the client wants to list dir_a
From Server : b.txt file1 file2 file3 lambda_a_file supp suppme test test.txt test_file 
Enter the string : list . #the client wants to list /root/partage/
From Server : admin.sh client client.c dir_a dir_b dir_c get_things get_things.c lambda_a.sh l��ü�
Enter the string : close # the client terminates the exchange
Client Exit...

```

In the example, and referring to our test script, it is possible to list the contents of a directory (here dir_a, with the `admin` user).

---

🔒 Secure your systems, shield your data 🔒