// ignore_for_file: prefer_const_constructors
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'dart:io';
import 'utils.dart';
import 'package:image/image.dart' as imgpack;

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
      Fluttertoast.showToast(msg: '找不到相机');
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
      body: Column(children: [
        Container(
          // padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
          height: 400,
          width: 400,
          child: controller == null
              ? Center(child: Text("Loading Camera..."))
              : !controller!.value.isInitialized
                  ? Center(
                      child: CircularProgressIndicator(),
                    )
                  : Center(
                      child: CameraPreview(controller!),
                    )
        ),
        takepic(),
      ])
    );
  }

  void autocam() async {
    List<String> photos = [];
    List<http.MultipartFile> newList = [];
    if (controller != null) {
      //check if contrller is not null
      if (controller!.value.isInitialized) {
        //check if controller is initialized
        for (var i = 0; i < 10; i++) {
          image = await controller!.takePicture(); //capture image
          File file = File(image!.path);
          // Read a jpeg image from file path
          imgpack.Image? resizedImage =
              imgpack.decodeImage(file.readAsBytesSync());
          // Resize the image
          resizedImage =
              imgpack.copyResize(resizedImage!, width: 250, height: 250);
          // Save the resize image
          file = file
            ..writeAsBytesSync(imgpack.encodeJpg(resizedImage, quality: 100));
          photos.add(file.path);
        }
        for (var img in photos) {
          if (img != "") {
            var multipartFile = await http.MultipartFile.fromPath(
              'images',
              File(img).path,
              filename: img.split('/').last,
            );
            newList.add(multipartFile);
          }
        }
        int? userPhone = loginUserPreference!.getInt("phone");
        String reqUrl = API.userUrl;
        var request =
            http.MultipartRequest('put', Uri.parse("$reqUrl/$userPhone"));
        request.files.addAll(newList);
        await request.send();
        setState(() {});
      }
    }
  }

  FloatingActionButton takepic() {
    return FloatingActionButton(
      onPressed: () async {
        autocam();
      },
      child: const Icon(Icons.camera_alt),
    );
  }
}
