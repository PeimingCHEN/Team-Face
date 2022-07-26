// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:fluttertoast/fluttertoast.dart';

class SignUpPage extends StatefulWidget {
  const SignUpPage({Key? key}) : super(key: key);
  static String tag = 'signup-page';
  @override
  State<SignUpPage> createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  TextEditingController invitationController = TextEditingController();
  TextEditingController nameController = TextEditingController();
  TextEditingController emailController = TextEditingController();
  TextEditingController phoneController = TextEditingController();
  TextEditingController passwordController = TextEditingController();
  TextEditingController repasswordController = TextEditingController();
  bool _isLoading = false;

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
                    controller: invitationController,
                    decoration: InputDecoration(
                      hintText: "请输入邀请码",
                    ),
                  ),
                  SizedBox(height: 15),
                  TextFormField(
                    controller: nameController,
                    decoration: InputDecoration(
                      hintText: "请输入姓名",
                    ),
                  ),
                  SizedBox(height: 15),
                  TextFormField(
                    controller: emailController,
                    decoration: InputDecoration(
                      hintText: "请输入邮箱",
                    ),
                  ),
                  SizedBox(height: 15),
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
                      hintText: "请输入登录密码",
                    ),
                    obscureText: true,
                  ),
                  SizedBox(height: 15),
                  TextFormField(
                    controller: repasswordController,
                    decoration: InputDecoration(
                      hintText: "请再次输入登录密码",
                    ),
                    obscureText: true,
                  ),
                  SizedBox(height: 30),
                  signupBTN()
                ],
              ),
          )
        ),
    );
  }

  signup(
    String invitation, String name, String email, String phone,
    String password, String repassword)
    async {
      if (password != repassword) {
        setState(() {
          _isLoading = false;
          Fluttertoast.showToast(msg: '密码不一致');
        });
      } else {
        Map data = {
          'organization': invitation,
          'name': name,
          'email': email,
          'phone': phone,
          'password': password,
          };
        var response = await http.post(
          Uri.parse("http://10.0.2.2:8000/accounts/user"),
          body: data);
        if (response.statusCode == 201) {
          setState(() {
            _isLoading = false;
            Fluttertoast.showToast(msg: '注册成功');
            Navigator.pop(context);
          });
        } else {
          setState(() {
            _isLoading = false;
            Fluttertoast.showToast(msg: '注册失败');
          });
        }
      }
    }

  ElevatedButton signupBTN() {
    return ElevatedButton(
        onPressed: () {
          setState(() {
            _isLoading = true;
          });
          signup(
            invitationController.text,
            nameController.text,
            emailController.text,
            phoneController.text, 
            passwordController.text,
            repasswordController.text);
        },
        autofocus: true,
        style: ButtonStyle(
          backgroundColor: MaterialStateProperty.all(Colors.indigo),
          minimumSize: MaterialStateProperty.all(Size(200, 50)),
        ),
        child: Text(
          "注册",
          style: TextStyle(fontSize: 20),
        ));
  }
}
