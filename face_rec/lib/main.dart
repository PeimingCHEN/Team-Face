// ignore_for_file: library_private_types_in_public_api
// ignore_for_file: prefer_const_constructors

import 'package:flutter/material.dart';
import 'package:face_rec/login_page.dart';
import 'package:face_rec/signup_page.dart';
import 'package:face_rec/home_page.dart';
import 'package:face_rec/setup_page.dart';

void main() {
  // Ensure that plugin services are initialized so that `availableCameras()`
  // can be called before `runApp()`
  WidgetsFlutterBinding.ensureInitialized();
  // Obtain a list of the available cameras on the device.
  // final cameras = await availableCameras();
  //get the front camera and do what you want
  // final frontCam = cameras[1];
  runApp(const MyApp());
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
        SetUpPage.tag: (context) => const SetUpPage(),
      },
    );
  }
}
