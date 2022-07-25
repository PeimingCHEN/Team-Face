from django.db import models
from django.core.files.storage import FileSystemStorage
from backend.settings import MEDIA_ROOT, MEDIA_URL

# INVITATION_CODE_TYPE = [
#     ('限时', '限时'),
#     ('一次性', '一次性')
# ]

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


# class InvitationCode(models.Model):

#     code = models.CharField(primary_key=True,max_length=128)
#     organization = models.ForeignKey(Organization,on_delete=models.CASCADE)
#     code_type = models.CharField(max_length=128,choices=INVITATION_CODE_TYPE,default="一次性")
#     property = models.IntegerField(default=1)
#     use_status = models.BinaryField(default=False)
#     created_time = models.DateField(auto_now_add=True,
#                                     verbose_name=u'注册时间')
#     update_time = models.DateField(auto_now=True,
#                                     verbose_name=u'更新时间')


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

class UserFaceImage(models.Model):
    class Meta:
        verbose_name = '用户头像照片'
        verbose_name_plural = '用户头像照片'
    user = models.ForeignKey(User,
                             on_delete=models.CASCADE,
                             related_name='images')
    image = models.ImageField(
        upload_to='user/', storage=image_storage, verbose_name='头像照片')
