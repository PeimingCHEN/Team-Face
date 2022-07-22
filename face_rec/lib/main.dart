// ignore_for_file: library_private_types_in_public_api
// ignore_for_file: prefer_const_constructors

import 'package:flutter/material.dart';
import 'package:face_rec/login_page.dart';
import 'package:face_rec/signup_page.dart';
import 'package:face_rec/home_page.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: LoginPage(),
      routes: <String, WidgetBuilder>{
        LoginPage.tag: (context) => const LoginPage(),
        SignUpPage.tag: (context) => const SignUpPage(),
        HomePage.tag: (context) => const HomePage(),
      },
    );
  }
}
