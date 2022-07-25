// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';

class SignUpPage extends StatefulWidget {
  const SignUpPage({Key? key}) : super(key: key);
  static String tag = 'signup-page';
  @override
  State<SignUpPage> createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.indigo,
        title: Text("用户注册"),
      ),
      body: Container(
        padding: EdgeInsets.symmetric(horizontal: 20, vertical: 30),
        height: MediaQuery.of(context).size.height,
        width: MediaQuery.of(context).size.width,
        child: Form(
          child: ListView(
            children: [
              const Text(
                  "Team Face",
                  style: TextStyle(
                    fontSize: 40,
                    fontStyle: FontStyle.italic,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              SizedBox(height: 10),
              TextFormField(
                decoration: InputDecoration(
                  hintText: "请输入公司码",
                ),
              ),
              SizedBox(height: 15),
              TextFormField(
                decoration: InputDecoration(
                  hintText: "请输入姓名",
                ),
              ),
              SizedBox(height: 15),
              TextFormField(
                decoration: InputDecoration(
                  hintText: "请输入邮箱",
                ),
              ),
              SizedBox(height: 15),
              TextFormField(
                decoration: InputDecoration(
                  hintText: "请输入手机号码",
                ),
              ),
              SizedBox(height: 15),
              TextFormField(
                decoration: InputDecoration(
                  hintText: "请输入登录密码",
                ),
              ),
              SizedBox(height: 15),
              TextFormField(
                decoration: InputDecoration(
                  hintText: "请再次输入登录密码",
                ),
              ),
              SizedBox(height: 30),
              ElevatedButton(
                  onPressed: () {},
                  autofocus: true,
                  style: ButtonStyle(
                      //设置按钮的颜色
                      backgroundColor:MaterialStateProperty.all(Colors.indigo),
                      //设置按钮的大小
                      minimumSize: MaterialStateProperty.all(Size(200, 50)),
                  ),
                  child: const Text(
                    "注册",
                    style: TextStyle(fontSize: 20),
                  )
              ),
            ],
          ),
        )
      ),
    );
  }
}
