
from django.db import models
from django.core.files.storage import FileSystemStorage
from backend.settings import MEDIA_ROOT, MEDIA_URL

INVITATION_CODE_TYPE = [
    ('限时', '限时'),
    ('一次性', '一次性')
]

image_storage = FileSystemStorage(location=MEDIA_ROOT,
                                    base_url=MEDIA_URL)

# Create your models here.
class Organization(models.Model):

    class Meta:
        verbose_name = '组织'
        verbose_name_plural = '组织'

    name = models.CharField(unique=True,max_length=128,primary_key=True)
    organization_code = models.CharField(verbose_name='组织代码',max_length=128,unique=True,blank=False)
    created_time = models.DateField(auto_now_add=True,
                                    verbose_name=u'注册时间')
    update_time = models.DateField(auto_now=True,
                                    verbose_name=u'更新时间')


class InvitationCode(models.Model):

    code = models.CharField(unique=True,primary_key=True,max_length=128)
    organization = models.ForeignKey(Organization,on_delete=models.CASCADE)
    code_type = models.CharField(max_length=128,choices=INVITATION_CODE_TYPE,default="一次性")
    property = models.IntegerField(default=1)
    use_status = models.BinaryField(default=False)
    created_time = models.DateField(auto_now_add=True,
                                    verbose_name=u'注册时间')
    update_time = models.DateField(auto_now=True,
                                    verbose_name=u'更新时间')


class User(models.Model):

    class Meta:
        verbose_name = '用户'
        verbose_name_plural = '用户'

    organization = models.ForeignKey(Organization,on_delete=models.CASCADE)
    name = models.CharField(unique=True,max_length=128)
    email = models.CharField(max_length=128)
    password = models.CharField(max_length=128)
    phone = models.BigIntegerField(blank=True, null=True,unique=True,
                                    verbose_name=u'联系电话')
    register_time = models.DateField(auto_now_add=True,
                                    verbose_name=u'注册时间')
    update_time = models.DateField(auto_now=True,
                                    verbose_name=u'更新时间')

    def __str__(self):
        return self.name

class UserFaceImage(models.Model):
    class Meta:
        verbose_name = ' 用户头像照片'
        verbose_name_plural = '用户头像照片'
    user = models.ForeignKey(User,blank=True, null=True,
                             on_delete=models.SET_NULL, 
                             verbose_name='用户')
    image = models.ImageField(upload_to='user/',
                              storage=image_storage,verbose_name='头像照片')





