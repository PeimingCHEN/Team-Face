// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
// import 'package:face_rec/login_page.dart';
import 'package:camera/camera.dart';
import 'facerec_page.dart';
import 'package:shared_preferences/shared_preferences.dart';

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);
  static String tag = 'home-page';
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late SharedPreferences sharedPreferences;

  @override
  void initState() {
    super.initState();
    checkLoginStatus();
  }

  checkLoginStatus() async {
    sharedPreferences = await SharedPreferences.getInstance();
    if (sharedPreferences.getInt("phone") == null) {
      if (!mounted) return;
      Navigator.pop(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.indigo,
        automaticallyImplyLeading: false, //关闭左侧退出箭头
        centerTitle: true,
        title: Text(
          "Face Rec",
          style: TextStyle(
            fontStyle: FontStyle.italic,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Container(
        padding: EdgeInsets.symmetric(horizontal: 20, vertical: 80),
        height: MediaQuery.of(context).size.height,
        width: MediaQuery.of(context).size.width,
        child: ListView(
          children: [
            ElevatedButton(
                onPressed: () {},
                autofocus: true,
                style: ButtonStyle(
                  //设置按钮的颜色
                  backgroundColor: MaterialStateProperty.all(Colors.indigo),
                  //设置按钮的大小
                  minimumSize: MaterialStateProperty.all(Size(200, 50)),
                ),
                child: const Text(
                  "设置",
                  style: TextStyle(fontSize: 20),
                )),
            SizedBox(height: 15),
            ElevatedButton(
                onPressed: () async {
                  await availableCameras().then((value) => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => FaceRecPage(
                          camera: value[1],
                        ),
                      ))); //点击跳转人脸识别界面
                },
                autofocus: true,
                style: ButtonStyle(
                  //设置按钮的颜色
                  backgroundColor: MaterialStateProperty.all(Colors.indigo),
                  //设置按钮的大小
                  minimumSize: MaterialStateProperty.all(Size(200, 50)),
                ),
                child: const Text(
                  "人脸识别",
                  style: TextStyle(fontSize: 20),
                )),
            SizedBox(height: 15),
            signoutBTN()
          ],
        ),
      ),
    );
  }

  ElevatedButton signoutBTN() {
    return ElevatedButton(
        onPressed: () {
          sharedPreferences.clear();
          Navigator.pop(context);
        },
        autofocus: true,
        style: ButtonStyle(
          backgroundColor: MaterialStateProperty.all(Colors.indigo),
          minimumSize: MaterialStateProperty.all(Size(200, 50)),
        ),
        child: Text(
          "退出",
          style: TextStyle(fontSize: 20),
        ));
  }
}
