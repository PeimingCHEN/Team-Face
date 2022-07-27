// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:io';
import 'utils.dart';

class SetUpPage extends StatefulWidget {
  const SetUpPage({Key? key}) : super(key: key);
  static String tag = 'setup-page';

  @override
  State<SetUpPage> createState() => _SetUpPageState();
}

class _SetUpPageState extends State<SetUpPage> {
  List<CameraDescription>? cameras; //list out the camera available
  CameraController? controller; //controller for camera
  XFile? image; //for captured image
  SharedPreferences? loginUserPreference; //get login user info

  @override
  void initState() {
    fetchUser();
    loadCamera();
    super.initState();
  }

  loadCamera() async {
    cameras = await availableCameras();
    if (cameras != null) {
      controller = CameraController(cameras![1], ResolutionPreset.medium);
      //cameras[0] = first camera, change to 1 to another camera

      controller!.initialize().then((_) {
        if (!mounted) {
          return;
        }
        setState(() {});
      });
    } else {
      print("NO any camera found");
    }
  }

  fetchUser() async {
    loginUserPreference = await SharedPreferences.getInstance();
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    controller!.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.indigo,
        centerTitle: true,
        title: Text(
          "Team Face",
          style: TextStyle(
            fontStyle: FontStyle.italic,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Container(
          child: Column(children: [
        Container(
          padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
          height: 300,
          width: 300,
          child: controller == null
              ? Center(child: Text("Loading Camera..."))
              : !controller!.value.isInitialized
                  ? Center(
                      child: CircularProgressIndicator(),
                    )
                  : RotationTransition(
                      turns: AlwaysStoppedAnimation(90 / 360),
                      child: CameraPreview(controller!),
                    ),
        ),
        takepic(),
        Container(
          //show captured image
          padding: EdgeInsets.all(30),
          child: image == null
              ? Text("No image captured")
              : Image.file(
                  File(image!.path),
                  height: 100,
                ),
          //display captured image
        )
      ])),
    );
  }

  FloatingActionButton takepic() {
    return FloatingActionButton(
      onPressed: () async {
        try {
          if (controller != null) {
            //check if contrller is not null
            if (controller!.value.isInitialized) {
              //check if controller is initialized
              image = await controller!.takePicture(); //capture image
              File file = File(image!.path);
              int? userPhone = loginUserPreference!.getInt("phone");
              String reqUrl = API.userUrl;
              var request = http.MultipartRequest(
                  'put', Uri.parse("$reqUrl/$userPhone"));
              // request.fields.addAll(
              //   {'user': 'pm'}
              // ); //后续可以在这里加其他修改的信息，如密码
              request.files.add(http.MultipartFile.fromBytes(
                  'images', File(file.path).readAsBytesSync(),
                  filename: file.path));
              var res = await request.send();
              setState(() {});
            }
          }
        } catch (e) {
          print(e); //show error
        }
      },
      child: const Icon(Icons.camera_alt),
    );
  }
}
