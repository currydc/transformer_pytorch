# -*- coding: utf-8 -*-
import pkg_resources

try:
    # 获取所有已安装的包
    installed_packages = [(d.project_name, d.version) for d in pkg_resources.working_set]

    # 按包名排序
    installed_packages.sort(key=lambda x: x[0].lower())

    # 写入文件
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        for package, version in installed_packages:
            f.write(f'{package}=={version}\n')

    print(f"成功导出 {len(installed_packages)} 个包到 requirements.txt")

except Exception as e:
    print(f"导出失败: {e}")
