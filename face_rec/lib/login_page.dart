// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'dart:convert';
import 'signup_page.dart';
import 'home_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({Key? key}) : super(key: key);
  static String tag = 'login-page';
  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  TextEditingController phoneController = TextEditingController();
  TextEditingController passwordController = TextEditingController();
  bool _isLoading = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
          padding: EdgeInsets.symmetric(horizontal: 20, vertical: 100),
          height: MediaQuery.of(context).size.height,
          width: MediaQuery.of(context).size.width,
          child: Form(
            child: _isLoading
                ? Center(child: CircularProgressIndicator())
                : ListView(
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
                        controller: phoneController,
                        keyboardType: TextInputType.number,
                        decoration: InputDecoration(
                          hintText: "请输入手机号码",
                        ),
                      ),
                      SizedBox(height: 15),
                      TextFormField(
                        controller: passwordController,
                        decoration: InputDecoration(
                          hintText: '请输入登录密码',
                        ),
                        obscureText: true,
                      ),
                      SizedBox(height: 30),
                      loginBTN(),
                      SizedBox(height: 15),
                      ElevatedButton(
                          onPressed: () {
                            Navigator.of(context)
                                .pushNamed(SignUpPage.tag); //点击跳转注册界面
                          },
                          autofocus: true,
                          style: ButtonStyle(
                            backgroundColor:
                                MaterialStateProperty.all(Colors.indigo),
                            minimumSize:
                                MaterialStateProperty.all(Size(200, 50)),
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

  login(String phone, String password) async {
    Map data = {'password': password};
    SharedPreferences sharedPreferences = await SharedPreferences.getInstance();
    var response = await http.post(
        Uri.parse("http://10.0.2.2:8000/accounts/user/$phone"),
        body: data);
    if (response.statusCode == 200) {
      var jsonResponse = json.decode(response.body);
      setState(() {
        _isLoading = false;
        sharedPreferences.setInt("phone", jsonResponse['phone']);
        Navigator.of(context).pushNamed(HomePage.tag);
      });
    } else {
      setState(() {
        _isLoading = false;
        Fluttertoast.showToast(msg: '登录失败!');
      });
    }
  }

  ElevatedButton loginBTN() {
    return ElevatedButton(
        onPressed: () {
          setState(() {
            _isLoading = true;
          });
          login(phoneController.text, passwordController.text);
          phoneController.clear();
          passwordController.clear();
        },
        autofocus: true,
        style: ButtonStyle(
          backgroundColor: MaterialStateProperty.all(Colors.indigo),
          minimumSize: MaterialStateProperty.all(Size(200, 50)),
        ),
        child: Text(
          "登录",
          style: TextStyle(fontSize: 20),
        ));
  }
}
