from sre_parse import State
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import Http404
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes
import json;
from accounts.serializers import (
    OrganizationListSerializer,
    OrganizationSerializer,
    UserListSerializer,
    UserSerializer,
    UserTestImageSerializer
)
from accounts.models import (
    Organization,
    User,
    UserTestImage,
    FaceImage_delete,
    TestImage_delete
)
import fr_algorithms.siamese as siamese


# @permission_classes([IsAuthenticated])
class organization_list_apiview(APIView):
    def get(self, request, format=None):
        organizations = Organization.objects.all()
        serializer = OrganizationListSerializer(organizations, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = OrganizationListSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                                status=status.HTTP_201_CREATED)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)


# @permission_classes([IsAuthenticated])
class organization_apiview(APIView):
    def get_object(self, name):
        try:
            return Organization.objects.get(name=name)
        except Organization.DoesNotExist:
            raise Http404

    def get(self, request, name, format=None):
        organization = self.get_object(name)
        serializer = OrganizationSerializer(organization)
        return Response(serializer.data)

    def put(self, request, name, format=None):
        organization = self.get_object(name)
        serializer = OrganizationSerializer(organization,
                                        data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, name, format=None):
        organization = self.get_object(name)
        organization.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)



class user_list_apiview(APIView):
    def get(self, request, format=None):
        users = User.objects.all()
        serializer = UserListSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        # 用户注册功能
        # 获得填写的邀请码
        invitation_code = request.data.get('organization')
        try:
            # 根据邀请码获取组织
            organization = Organization.objects.get(invitation_code=invitation_code)
        except Organization.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        data=request.data.copy()
        data['organization'] = organization.name
        serializer = UserListSerializer(data=data)
        # 注册用户
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                                status=status.HTTP_201_CREATED)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)


# @permission_classes([IsAuthenticated])
class user_apiview(APIView):
    def get_object(self, phone):
        try:
            return User.objects.get(phone=phone)
        except User.DoesNotExist:
            raise Http404

    def get(self, request, phone, format=None):
        user = self.get_object(phone)
        serializer = UserSerializer(user)
        return Response(serializer.data)

    def post(self, request, phone, format=None):
        # 用户登录功能
        user = self.get_object(phone)
        serializer = UserSerializer(user)
        # 获得response中密码
        password = request.data.get('password')
        if user.password == password:
            # 匹配成功则登录成功
            return Response(serializer.data)
        else:
            return Response(status=status.HTTP_404_NOT_FOUND)
    
    def put(self, request, phone, format=None):
        # 用户设置功能，包括上传anchor，及后续修改密码等操作
        user = self.get_object(phone)
        serializer = UserSerializer(user,
                                    data=request.data)
        if serializer.is_valid():
            imagelist = dict((request.data).lists())['images']
            old_images = user.images.all()
            for image in old_images:
                FaceImage_delete(instance=image)
                image.delete()

            for image in imagelist:
                user_img = user.images.create()
                user_img.image.save(image.name, image)
            serializer.save()
            # train
            # siamese.copy_face_images(str(user.phone))
            # siamese.update_model_with_new_training_data(str(user.phone))
            return Response(status=status.HTTP_200_OK)
        return Response(serializer.errors,
                        status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, phone, format=None):
        user = self.get_object(phone)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class test_img_apiview(APIView):
    def get(self, request, format=None):
        imgs = UserTestImage.objects.all()
        serializer = UserTestImageSerializer(imgs, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        # 用户上传人脸识别测试图像
        # 获得上传图像用户的身份
        phone = request.data.get('phone')
        try:
            # 根据手机号获取用户
            user = User.objects.get(phone=phone)
        except User.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        test_image_req = request.FILES['test_images']
        old_images = user.test.all()
        for image in old_images:
            TestImage_delete(instance=image)
            image.delete()
        test_image = user.test.create()
        test_image.test_image.save(test_image_req.name, test_image_req)
        # test
        result = siamese.recognize_organization(str(phone))
        if result != 'unrecognized identity!':
            detect_user = User.objects.get(phone=int(result))
            result = detect_user.organization.name
        else:
            result = '无法识别身份'
        result='{"result":"'+result+'"}'
        return Response(json.loads(result), status=status.HTTP_200_OK)
