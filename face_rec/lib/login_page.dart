// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'signup_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({Key? key}) : super(key: key);
  static String tag = 'login-page';
  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
          padding: EdgeInsets.symmetric(horizontal: 20, vertical: 100),
          height: MediaQuery.of(context).size.height,
          width: MediaQuery.of(context).size.width,
          child: Form(
            child: ListView(
              children: [
                const Text(
                  "Face Rec",
                  style: TextStyle(
                    fontSize: 40,
                    fontStyle: FontStyle.italic,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 10),
                TextFormField(
                  decoration: InputDecoration(
                    hintText: "请输入手机号码",
                  ),
                ),
                SizedBox(height: 15),
                TextFormField(
                  decoration: InputDecoration(
                    hintText: '请输入登录密码',
                  ),
                  obscureText: true,
                ),
                SizedBox(height: 30),
                ElevatedButton(
                    onPressed: () {},
                    autofocus: true,
                    style: ButtonStyle(
                        backgroundColor:MaterialStateProperty.all(Colors.indigo),
                        minimumSize: MaterialStateProperty.all(Size(200, 50)),
                      ),
                    child: Text(
                      "登录",
                      style: TextStyle(fontSize: 20),
                    )),
                SizedBox(height: 15),
                ElevatedButton(
                    onPressed: () {
                      Navigator.of(context).pushNamed(SignUpPage.tag); //点击跳转注册界面
                    },
                    autofocus: true,
                    style: ButtonStyle(
                        backgroundColor:MaterialStateProperty.all(Colors.indigo),
                        minimumSize: MaterialStateProperty.all(Size(200, 50)),
                      ),
                    child: const Text(
                      "注册",
                      style: TextStyle(fontSize: 20),
                    )),
              ],
            ),
          )),
    );
  }
}
