from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from django.core.files.storage import FileSystemStorage
from backend.settings import MEDIA_ROOT, MEDIA_URL


image_storage = FileSystemStorage(location=MEDIA_ROOT,
                                    base_url=MEDIA_URL)

# Create your models here.
class Organization(models.Model):

    class Meta:
        verbose_name = '组织'
        verbose_name_plural = '组织'

    name = models.CharField(
        max_length=128, primary_key=True, verbose_name='公司名称')
    organization_code = models.CharField(
        max_length=128, unique=True, verbose_name='组织代码')
    invitation_code = models.CharField(
        max_length=128, unique=True, verbose_name='邀请码')
    created_time = models.DateField(
        auto_now_add=True, verbose_name='注册时间')
    update_time = models.DateField(
        auto_now=True, verbose_name='更新时间')


class User(models.Model):

    class Meta:
        verbose_name = '用户'
        verbose_name_plural = '用户'

    organization = models.ForeignKey(Organization,
                                    related_name='members',
                                    on_delete=models.CASCADE)
    name = models.CharField(
        max_length=128, verbose_name='姓名')
    email = models.CharField(
        max_length=128, verbose_name='邮箱')
    password = models.CharField(
        max_length=128, verbose_name='密码')
    phone = models.BigIntegerField(
        primary_key=True, verbose_name='联系电话')
    register_time = models.DateField(
        auto_now_add=True, verbose_name='注册时间')
    update_time = models.DateField(
        auto_now=True, verbose_name='更新时间')

    def __str__(self):
        return self.name


def user_directory_path(instance, filename):
    return 'user/{}/anchor/{}'.format(instance.user.phone, filename)

def test_directory_path(instance, filename):
    return 'user/{}/test/{}'.format(instance.user.phone, filename)


class UserFaceImage(models.Model):
    class Meta:
        verbose_name = '用户头像照片'
        verbose_name_plural = '用户头像照片'


    user = models.ForeignKey(User,
                             on_delete=models.CASCADE,
                             related_name='images')
    image = models.ImageField(
        upload_to=user_directory_path, storage=image_storage, verbose_name='头像照片')

@receiver(pre_delete, sender=UserFaceImage) #sender=你要删除或修改文件字段所在的类**
def FaceImage_delete(instance, **kwargs):       #函数名随意
    instance.image.delete(False) #file是保存文件或图片的字段名**


class UserTestImage(models.Model):
    class Meta:
        verbose_name = '用户测试照片'
        verbose_name_plural = '用户测试照片'


    user = models.ForeignKey(User,
                             on_delete=models.CASCADE,
                             related_name='test')
    test_image = models.ImageField(
        upload_to=test_directory_path, storage=image_storage, verbose_name='测试照片')

@receiver(pre_delete, sender=UserTestImage) #sender=你要删除或修改文件字段所在的类**
def TestImage_delete(instance, **kwargs):       #函数名随意
    instance.test_image.delete(False) #file是保存文件或图片的字段名**
